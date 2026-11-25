import { BUILDINGS, getBuildingFootprint, getBuildingLevelMultiplier } from '../data/buildings.data';
import type { ResourceId } from '../data/resources.data';
import { getRoadProperties, ROADS, RoadType } from '../data/roads.data';
import { FeatureType, isMatureTreeFeature, TREE_SPECIES } from '../data/trees.data';
import type { ChunkManager } from '../core/ChunkManager';
import {
  BuildingInstance,
  ForestryWorkerAgent,
  LoggingTruckAgent,
  RoadVehicleAgent,
  RoadVehicleKind,
  TileFlag,
  TrainAgent,
  VehicleClass,
  WorldEngineState,
} from '../core/GameState';
import { CHUNK_SIZE, gridToChunk, localTileIndex, tileKey } from '../core/IsoMath';
import type { ForestrySystem } from './ForestrySystem';
import { PathfindingSystem } from './PathfindingSystem';
import { ResourceSystem } from './ResourceSystem';
import type { RoadNetworkSystem } from './RoadNetworkSystem';

export interface TransportDeliveryEvent {
  gx: number;
  gy: number;
  label: string;
  color: string;
}

export interface FactoryFleetSummary {
  factoryId: number;
  totalTrucks: number;
  availableTrucks: number;
  activeTrucks: number;
  travellingTrucks: number;
  loadingTrucks: number;
  waitingTrucks: number;
  returningTrucks: number;
  transportingCargoTrucks: number;
  destinations: string[];
}

export interface LogisticsRouteDiagnostic {
  id: string;
  category: 'vehicle_route' | 'forestry';
  severity: 'warning' | 'critical';
  title: string;
  detail: string;
  gx: number;
  gy: number;
}

const SAWMILL_IDS = new Set([
  'lumber_mill',
  'heavy_lumber_complex',
  'sawmill_mega',
  'plywood_factory',
  'timber_yard',
]);

const CARGO_FALLBACK_RESOURCES: ResourceId[] = ['wood', 'stone', 'wheat', 'bricks', 'iron_ingots', 'tools'];

function getVehicleClass(kind: RoadVehicleKind): VehicleClass {
  switch (kind) {
    case 'car':
    case 'taxi':
      return 'car';
    case 'delivery_van':
    case 'fish_van':
      return 'van';
    case 'bus':
      return 'bus';
    case 'emergency':
      return 'emergency';
    default:
      return 'heavy_truck';
  }
}

function getBaseMaxSpeed(kind: RoadVehicleKind): number {
  switch (kind) {
    case 'emergency':
      return 3.4;
    case 'car':
      return 2.95;
    case 'taxi':
      return 2.85;
    case 'delivery_van':
    case 'fish_van':
      return 2.55;
    case 'bus':
      return 2.25;
    case 'cargo_truck':
    case 'construction_truck':
      return 2.05;
    case 'container_truck':
      return 1.9;
    case 'tractor':
      return 1.6;
  }
}

function getLaneOffsetForTile(
  gx: number,
  gy: number,
  vehicleId: number,
  chunks: ChunkManager
): number {
  const chunk = chunks.getLoadedChunkForGrid(gx, gy);
  if (!chunk) return 0;
  const rt = chunk.roadType[localTileIndex(gx, gy)] as RoadType;
  if (rt === RoadType.None || rt === RoadType.Rail) return 0;
  const props = getRoadProperties(rt);
  if (props.totalLanes <= 1) return 0;
  const baseOffset =
    props.widthClass === 'highway'
      ? 0.24
      : props.widthClass === 'wide'
        ? 0.2
        : 0.16;
  const multiLaneShift = props.lanesPerDirection >= 2 ? (vehicleId % 2 === 0 ? -0.04 : 0.05) : 0;
  return baseOffset + multiLaneShift;
}

interface CachedForestZone {
  zoneKey: number;
  zx: number;
  zy: number;
  zGx: number;
  zGy: number;
  trees: Array<{ gx: number; gy: number; key: number; permitted: boolean }>;
  roadStops: Array<{ gx: number; gy: number; key: number }>;
}

interface ForestryDispatchOutcome {
  stopRoad?: { gx: number; gy: number };
  entrance?: { gx: number; gy: number };
  path?: Array<{ gx: number; gy: number }>;
  zoneKey?: number;
  zoneCenterGx?: number;
  zoneCenterGy?: number;
  matureCount?: number;
  reason?: string;
  diagnostic?: LogisticsRouteDiagnostic;
}

export class LogisticsAndTransportSystem {
  public readonly loggingTrucks: LoggingTruckAgent[] = [];
  public readonly trains: TrainAgent[] = [];
  public readonly roadVehicles: RoadVehicleAgent[] = [];

  private nextId = 1;
  private nextWorkerId = 1;
  private syncAccum = 0;
  private trafficSampleAccum = 0;
  private internalSimClockSec = 0;
  private cachedRailTiles: Array<{ gx: number; gy: number }> = [];
  private cachedRoadTiles: Array<{ gx: number; gy: number; unlocked?: boolean; clusterId?: number }> = [];
  private cachedRoadTilesByCluster = new Map<number, Array<{ gx: number; gy: number }>>();
  private cachedLockedRoadTiles: Array<{ gx: number; gy: number }> = [];
  private railSet = new Set<number>();
  private roadSet = new Set<number>();
  private cachedNetVer = -1;
  private harvestedToday = 0;
  private lastDayReset = -1;

  // Actual wood units carried per logging truck (while cargoTimber tracks exact 30..40 trees loaded)
  private pendingWoodByTruckId = new Map<number, number>();
  // Depleted forestry zone cooldowns (`zoneKey -> expiryTick`) so trucks rotate cleanly
  private depletedZones = new Map<number, number>();
  // Heavy-truck / forestry route issues observed during live dispatch
  private routeDiagnosticsMap = new Map<string, LogisticsRouteDiagnostic>();

  // Cached locked forestry work zones & negative factory dispatch backoff
  private cachedForestZones: CachedForestZone[] = [];
  private forestIndexTick = -999;
  private forestIndexNetVer = -1;
  private forestIndexDirty = true;
  private forestChopsSinceIndex = 0;
  private factoryDispatchBackoff = new Map<
    number,
    { netVer: number; untilSimSec: number; outcome: ForestryDispatchOutcome }
  >();

  public reset(): void {
    this.loggingTrucks.length = 0;
    this.trains.length = 0;
    this.roadVehicles.length = 0;
    this.cachedRailTiles = [];
    this.cachedRoadTiles = [];
    this.cachedRoadTilesByCluster.clear();
    this.cachedLockedRoadTiles = [];
    this.railSet.clear();
    this.roadSet.clear();
    this.cachedNetVer = -1;
    this.syncAccum = 0;
    this.trafficSampleAccum = 0;
    this.internalSimClockSec = 0;
    this.harvestedToday = 0;
    this.lastDayReset = -1;
    this.pendingWoodByTruckId.clear();
    this.depletedZones.clear();
    this.routeDiagnosticsMap.clear();
    this.cachedForestZones = [];
    this.forestIndexTick = -999;
    this.forestIndexNetVer = -1;
    this.forestIndexDirty = true;
    this.forestChopsSinceIndex = 0;
    this.factoryDispatchBackoff.clear();
  }

  public getRouteDiagnostics(): LogisticsRouteDiagnostic[] {
    return Array.from(this.routeDiagnosticsMap.values());
  }

  public getFactoryFleetSummary(factoryId: number): FactoryFleetSummary {
    const fleet = this.loggingTrucks.filter((t) => t.factoryId === factoryId);
    let availableTrucks = 0;
    let activeTrucks = 0;
    let travellingTrucks = 0;
    let loadingTrucks = 0;
    let waitingTrucks = 0;
    let returningTrucks = 0;
    let transportingCargoTrucks = 0;
    const destSet = new Set<string>();

    for (const t of fleet) {
      if (t.state === 'idle_at_factory') {
        if (t.workStatus === 'waiting') {
          waitingTrucks++;
        } else {
          availableTrucks++;
        }
      } else if (t.state === 'blocked') {
        waitingTrucks++;
      } else {
        activeTrucks++;
        if (t.state === 'to_forest') {
          travellingTrucks++;
        } else if (t.state === 'worker_chopping' || t.state === 'chopping') {
          loadingTrucks++;
        } else if (t.state === 'to_factory' || t.state === 'delivering_sawmill') {
          returningTrucks++;
        }
      }

      if (t.cargoTimber > 0) {
        transportingCargoTrucks++;
      }
      if (t.assignedDestination) {
        destSet.add(t.assignedDestination);
      }
    }

    return {
      factoryId,
      totalTrucks: fleet.length,
      availableTrucks,
      activeTrucks,
      travellingTrucks,
      loadingTrucks,
      waitingTrucks,
      returningTrucks,
      transportingCargoTrucks,
      destinations: Array.from(destSet),
    };
  }

  public update(
    dtSec: number,
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet?: RoadNetworkSystem,
    simTimeSec?: number,
    onEvent?: (ev: TransportDeliveryEvent) => void,
    forestry?: ForestrySystem
  ): void {
    this.internalSimClockSec += dtSec;
    const effectiveTimeSec = simTimeSec ?? this.internalSimClockSec;

    const dayKey = state.year * 100 + state.dayInSeason;
    if (this.lastDayReset !== dayKey) {
      this.lastDayReset = dayKey;
      this.harvestedToday = 0;
    }

    this.syncAccum += dtSec;
    if (
      this.syncAccum >= 2.0 ||
      (this.trains.length === 0 &&
        this.loggingTrucks.length === 0 &&
        this.roadVehicles.length === 0 &&
        this.syncAccum >= 0.4)
    ) {
      this.syncAccum = 0;
      if (roadNet) {
        roadNet.recomputeIfDirty(chunks, state);
      }
      this.rebuildNetworkCacheIfNeeded(chunks, roadNet);
      this.syncLoggingTrucks(state, chunks);
      this.syncTrains(state);
      this.syncRoadVehicles(state, chunks, roadNet);
    }

    this.stepLoggingTrucks(dtSec, state, chunks, roadNet, effectiveTimeSec, onEvent, forestry);
    this.stepTrains(dtSec);
    this.stepRoadVehicles(dtSec, state, chunks, roadNet, effectiveTimeSec, onEvent);

    if (roadNet) {
      this.trafficSampleAccum += dtSec;
      if (this.trafficSampleAccum >= 0.35) {
        this.trafficSampleAccum = 0;
        const tileVehicleCount = new Map<number, number>();
        const tileWaitingCount = new Map<number, number>();

        for (let i = 0; i < this.roadVehicles.length; i++) {
          const rv = this.roadVehicles[i];
          const key = tileKey(Math.round(rv.gx), Math.round(rv.gy));
          tileVehicleCount.set(key, (tileVehicleCount.get(key) ?? 0) + 1);
          if (rv.state === 'waiting_signal' || rv.state === 'waiting_traffic') {
            tileWaitingCount.set(key, (tileWaitingCount.get(key) ?? 0) + 1);
          }
        }
        for (let i = 0; i < this.loggingTrucks.length; i++) {
          const lt = this.loggingTrucks[i];
          if (!lt.onRoad) continue;
          const key = tileKey(Math.round(lt.gx), Math.round(lt.gy));
          tileVehicleCount.set(key, (tileVehicleCount.get(key) ?? 0) + 1);
          if ((lt.waitTimer ?? 0) > 0 || lt.workStatus === 'waiting') {
            tileWaitingCount.set(key, (tileWaitingCount.get(key) ?? 0) + 1);
          }
        }

        roadNet.recordTrafficSample(tileVehicleCount, tileWaitingCount, chunks, state);
      }
    }
  }

  private rebuildNetworkCacheIfNeeded(chunks: ChunkManager, roadNet?: RoadNetworkSystem): void {
    const targetVer = roadNet ? roadNet.networkVersion : -2;
    if (roadNet && this.cachedNetVer === targetVer) {
      return;
    }
    this.cachedNetVer = targetVer;
    this.cachedRailTiles = [];
    this.cachedRoadTiles = [];
    this.cachedRoadTilesByCluster.clear();
    this.cachedLockedRoadTiles = [];
    this.railSet.clear();
    this.roadSet.clear();

    if (roadNet) {
      const allRoads = roadNet.getCachedRoadTiles();
      for (let i = 0; i < allRoads.length; i++) {
        const entry = allRoads[i];
        if (entry.type === RoadType.Rail) {
          this.cachedRailTiles.push({ gx: entry.gx, gy: entry.gy });
          this.railSet.add(entry.key);
        } else {
          this.cachedRoadTiles.push(entry);
          this.roadSet.add(entry.key);
          if (!entry.unlocked) {
            this.cachedLockedRoadTiles.push({ gx: entry.gx, gy: entry.gy });
          }
          if (entry.clusterId > 0) {
            let list = this.cachedRoadTilesByCluster.get(entry.clusterId);
            if (!list) {
              list = [];
              this.cachedRoadTilesByCluster.set(entry.clusterId, list);
            }
            list.push(entry);
          }
        }
      }
      return;
    }

    for (const chunk of chunks.getAllLoadedChunks()) {
      const baseX = chunk.cx * CHUNK_SIZE;
      const baseY = chunk.cy * CHUNK_SIZE;
      for (let ly = 0; ly < CHUNK_SIZE; ly++) {
        const gy = baseY + ly;
        const rowOff = ly * CHUNK_SIZE;
        for (let lx = 0; lx < CHUNK_SIZE; lx++) {
          const i = rowOff + lx;
          const rt = chunk.roadType[i] as RoadType;
          if (rt === RoadType.None) continue;
          const gx = baseX + lx;
          const key = tileKey(gx, gy);
          if (rt === RoadType.Rail) {
            this.cachedRailTiles.push({ gx, gy });
            this.railSet.add(key);
          } else {
            const unlocked = (chunk.flags[i] & TileFlag.Unlocked) !== 0;
            this.cachedRoadTiles.push({ gx, gy, unlocked });
            this.roadSet.add(key);
            if (!unlocked) {
              this.cachedLockedRoadTiles.push({ gx, gy });
            }
          }
        }
      }
    }
  }

  private syncLoggingTrucks(state: WorldEngineState, chunks: ChunkManager): void {
    const factories = state.buildings.filter((b) => {
      const def = BUILDINGS[b.defId];
      return def?.isAutomatedLogging && b.status !== 'constructing' && b.enabled;
    });

    const activeFactoryIds = new Set(factories.map((f) => f.id));
    for (let i = this.loggingTrucks.length - 1; i >= 0; i--) {
      if (!activeFactoryIds.has(this.loggingTrucks[i].factoryId)) {
        this.pendingWoodByTruckId.delete(this.loggingTrucks[i].id);
        this.loggingTrucks.splice(i, 1);
      }
    }

    for (const factory of factories) {
      const def = BUILDINGS[factory.defId]!;
      const targetTrucks = def.truckCapacity ?? 6;
      const existing = this.loggingTrucks.filter((t) => t.factoryId === factory.id);
      const fp = getBuildingFootprint(def, factory.rotation ?? 0);
      const cx = factory.gx + fp.width * 0.5;
      const cy = factory.gy + fp.height * 0.5;

      // Seed locked forest permits nearby if none exist yet
      if (existing.length === 0) {
        let hasPermitted = false;
        outer: for (let dy = -12; dy <= 12; dy++) {
          for (let dx = -12; dx <= 12; dx++) {
            if (chunks.isLoggingPermitted(Math.round(cx) + dx, Math.round(cy) + dy)) {
              hasPermitted = true;
              break outer;
            }
          }
        }
        if (!hasPermitted) {
          chunks.autoPermitForestForFactory(Math.round(cx), Math.round(cy), 14);
        }
      }

      const entrances = PathfindingSystem.findBuildingEntrances(factory, chunks, 'vehicle');
      const spawnPt = entrances.length > 0 ? entrances[0] : { gx: cx, gy: cy };
      // Cargo capacity is 30..40 mature trees per trip depending on factory level (Lv1=32, Lv2=36, Lv3=40)
      const cargoCapacity = Math.min(40, Math.max(30, 32 + (factory.level - 1) * 4));

      while (existing.length < targetTrucks) {
        const truck: LoggingTruckAgent = {
          id: this.nextId++,
          factoryId: factory.id,
          gx: spawnPt.gx,
          gy: spawnPt.gy,
          prevGx: spawnPt.gx,
          prevGy: spawnPt.gy,
          nextGx: spawnPt.gx,
          nextGy: spawnPt.gy,
          progress: 1,
          targetGx: spawnPt.gx,
          targetGy: spawnPt.gy,
          state: 'idle_at_factory',
          workStatus: 'idle',
          cargoTimber: 0,
          cargoCapacity,
          treesHarvestedTrip: 0,
          harvestTimer: 0.25 + ((existing.length * 0.31) % 2.4),
          facing: 1,
          route: undefined,
          routeIdx: 0,
          routeVersion: 0,
          laneOffset: 0,
          speed: 0,
          vx: 0,
          vy: 0,
          waitTimer: 0,
          blockedSec: 0,
          onRoad: entrances.length > 0,
          workers: [],
          statusReason: entrances.length > 0 ? 'Ready for dispatch' : 'Blocked: Connect a road adjacent to factory',
        };
        this.loggingTrucks.push(truck);
        existing.push(truck);
      }
      for (const t of existing) {
        t.cargoCapacity = cargoCapacity;
      }
    }

    state.loggingStats ??= {
      harvestedTrees: 0,
      factoryTimberStock: 0,
      dailyTimberProduction: 0,
      activeTrucks: 0,
      permittedForestTiles: 0,
    };
    state.loggingStats.activeTrucks = this.loggingTrucks.filter((t) => t.state !== 'idle_at_factory').length;
    state.loggingStats.permittedForestTiles = chunks.countLoggingPermittedTiles();
    state.loggingStats.factoryTimberStock = Math.round(state.resources.wood ?? 0);
    state.loggingStats.dailyTimberProduction = this.harvestedToday;
  }

  private ensureForestryZoneIndex(
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet?: RoadNetworkSystem
  ): CachedForestZone[] {
    const netVer = roadNet ? roadNet.networkVersion : -2;
    if (
      !this.forestIndexDirty &&
      this.forestIndexNetVer === netVer &&
      this.forestIndexTick >= 0 &&
      state.tick - this.forestIndexTick < 15
    ) {
      return this.cachedForestZones;
    }

    this.rebuildNetworkCacheIfNeeded(chunks, roadNet);
    this.forestIndexDirty = false;
    this.forestChopsSinceIndex = 0;
    this.forestIndexTick = state.tick;
    this.forestIndexNetVer = netVer;

    // 1. Collect all mature locked trees into 8x8 zones (zero object wrappers, fast typed array scan)
    const zoneBuckets = new Map<number, CachedForestZone>();
    for (const chunk of chunks.getAllLoadedChunks()) {
      const baseX = chunk.cx * CHUNK_SIZE;
      const baseY = chunk.cy * CHUNK_SIZE;
      for (let ly = 0; ly < CHUNK_SIZE; ly++) {
        const gy = baseY + ly;
        const zy = Math.floor(gy / 8);
        const rowOff = ly * CHUNK_SIZE;
        for (let lx = 0; lx < CHUNK_SIZE; lx++) {
          const i = rowOff + lx;
          if ((chunk.flags[i] & TileFlag.Unlocked) !== 0) continue; // Locked land only
          const ft = chunk.featureType[i] as FeatureType;
          if (!isMatureTreeFeature(ft)) continue;

          const gx = baseX + lx;
          const zx = Math.floor(gx / 8);
          const zk = tileKey(zx, zy);
          let zone = zoneBuckets.get(zk);
          if (!zone) {
            zone = {
              zoneKey: zk,
              zx,
              zy,
              zGx: zx * 8 + 4,
              zGy: zy * 8 + 4,
              trees: [],
              roadStops: [],
            };
            zoneBuckets.set(zk, zone);
          }
          zone.trees.push({
            gx,
            gy,
            key: tileKey(gx, gy),
            permitted: (chunk.flags[i] & TileFlag.LoggingPermit) !== 0,
          });
        }
      }
    }

    if (zoneBuckets.size === 0) {
      this.cachedForestZones = [];
      return this.cachedForestZones;
    }

    // 2. Match locked road tiles against adjacent 8x8 forest zones (O(#lockedRoads * small window))
    if (this.cachedLockedRoadTiles.length > 0) {
      const seenStopsByZone = new Map<number, Set<number>>();
      for (let i = 0; i < this.cachedLockedRoadTiles.length; i++) {
        const road = this.cachedLockedRoadTiles[i];
        const rzx = Math.floor(road.gx / 8);
        const rzy = Math.floor(road.gy / 8);
        const rKey = tileKey(road.gx, road.gy);

        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const zk = tileKey(rzx + dx, rzy + dy);
            const zone = zoneBuckets.get(zk);
            if (!zone) continue;

            let seen = seenStopsByZone.get(zk);
            if (seen && seen.has(rKey)) continue;

            // Verify at least one mature tree in this zone is within worker walking Manhattan distance <= 7
            let touchesTree = false;
            for (let ti = 0; ti < zone.trees.length; ti++) {
              const t = zone.trees[ti];
              const adx = Math.abs(road.gx - t.gx);
              const ady = Math.abs(road.gy - t.gy);
              if (adx <= 6 && ady <= 6 && adx + ady <= 7) {
                touchesTree = true;
                break;
              }
            }
            if (touchesTree) {
              if (!seen) {
                seen = new Set<number>();
                seenStopsByZone.set(zk, seen);
              }
              seen.add(rKey);
              zone.roadStops.push({ gx: road.gx, gy: road.gy, key: rKey });
            }
          }
        }
      }

      for (const zone of zoneBuckets.values()) {
        if (zone.roadStops.length > 1) {
          zone.roadStops.sort(
            (a, b) => Math.hypot(a.gx - zone.zGx, a.gy - zone.zGy) - Math.hypot(b.gx - zone.zGx, b.gy - zone.zGy)
          );
        }
      }
    }

    this.cachedForestZones = Array.from(zoneBuckets.values());
    return this.cachedForestZones;
  }

  /**
   * Evaluates LOCKED-land forestry work zones connected via an active Connector + road network.
   * Tracks mature tree density, avoids depleted zones and over-crowded sectors, and checks Connector bridging.
   */
  private findForestryWorkZoneAndRoute(
    factory: BuildingInstance,
    originRoad: { gx: number; gy: number } | undefined,
    state: WorldEngineState,
    chunks: ChunkManager,
    reservedTrees: Set<number>,
    activeZoneCounts: Map<number, number>,
    roadNet: RoadNetworkSystem | undefined,
    simTimeSec: number,
    cachedEntrances?: Array<{ gx: number; gy: number }>
  ): ForestryDispatchOutcome {
    const def = BUILDINGS[factory.defId];
    const fp = def ? getBuildingFootprint(def, factory.rotation ?? 0) : { width: 5, height: 5 };
    const cx = Math.round(factory.gx + fp.width * 0.5);
    const cy = Math.round(factory.gy + fp.height * 0.5);

    const entrances = originRoad
      ? [originRoad]
      : (cachedEntrances ?? PathfindingSystem.findBuildingEntrances(factory, chunks, 'vehicle'));
    if (entrances.length === 0) {
      return {
        reason: 'Blocked: Connect a drivable road to the Logging Factory',
        diagnostic: {
          id: `forestry_noroad_${factory.id}`,
          category: 'forestry',
          severity: 'critical',
          title: 'Logging Factory Disconnected',
          detail: 'Connect a drivable road next to the Automated Logging Factory.',
          gx: cx,
          gy: cy,
        },
      };
    }

    const netVer = roadNet ? roadNet.networkVersion : -2;
    if (!originRoad) {
      const backoff = this.factoryDispatchBackoff.get(factory.id);
      if (backoff && backoff.netVer === netVer && simTimeSec < backoff.untilSimSec) {
        return backoff.outcome;
      }
    }

    const cachedZones = this.ensureForestryZoneIndex(state, chunks, roadNet);
    if (cachedZones.length === 0) {
      return { reason: 'Waiting for locked-land mature trees to regenerate' };
    }

    const hasExplicitPermits = (state.loggingStats?.permittedForestTiles ?? 0) > 0 || chunks.countLoggingPermittedTiles() > 0;
    const candidateZones: Array<{
      zone: CachedForestZone;
      availCount: number;
      score: number;
    }> = [];
    let anyZoneWithinRange = false;

    for (let i = 0; i < cachedZones.length; i++) {
      const zone = cachedZones[i];
      if (Math.abs(zone.zGx - cx) > 52 || Math.abs(zone.zGy - cy) > 52) continue;
      const cooldownUntil = this.depletedZones.get(zone.zoneKey) ?? 0;
      if (state.tick < cooldownUntil) continue;

      let availCount = 0;
      let permittedCount = 0;
      for (let ti = 0; ti < zone.trees.length; ti++) {
        const t = zone.trees[ti];
        if (Math.abs(t.gx - cx) > 48 || Math.abs(t.gy - cy) > 48) continue;
        if (reservedTrees.has(t.key)) continue;
        // Skip tiles already chopped since index built
        const c = chunks.getLoadedChunkForGrid(t.gx, t.gy);
        if (!c || !isMatureTreeFeature(c.featureType[localTileIndex(t.gx, t.gy)] as FeatureType)) continue;
        availCount++;
        if (t.permitted) permittedCount++;
      }

      if (availCount === 0) continue;
      anyZoneWithinRange = true;

      const truckCrowd = activeZoneCounts.get(zone.zoneKey) ?? 0;
      if (truckCrowd >= 2 && cachedZones.length > 1) continue;
      if (availCount < 3 && permittedCount === 0 && cachedZones.length > 2) continue;
      if (zone.roadStops.length === 0) continue;

      const dist = Math.hypot(zone.zGx - cx, zone.zGy - cy);
      const permitBonus = hasExplicitPermits ? permittedCount * 4.0 : 0;
      const score = availCount * 3.0 + permitBonus - dist * 0.45 - truckCrowd * 24;
      candidateZones.push({ zone, availCount, score });
    }

    if (!anyZoneWithinRange) {
      return { reason: 'Waiting for locked-land mature trees to regenerate' };
    }

    if (candidateZones.length === 0) {
      const outcome: ForestryDispatchOutcome = {
        reason: 'Blocked: Extend a locked-area road + Connector near mature forest stands',
        diagnostic: {
          id: `forestry_nocolroad_${factory.id}`,
          category: 'forestry',
          severity: 'warning',
          title: 'Forestry Zones Lack Road Access',
          detail: 'Mature forest zones in locked land have no adjacent locked road within worker walking range.',
          gx: cx,
          gy: cy,
        },
      };
      if (!originRoad) {
        this.factoryDispatchBackoff.set(factory.id, { netVer, untilSimSec: simTimeSec + 2.2, outcome });
      }
      return outcome;
    }

    candidateZones.sort((a, b) => b.score - a.score);
    const congestionMap = roadNet?.getTileCongestionMap();
    const entranceClusters = new Set<number>();
    if (roadNet) {
      for (let i = 0; i < entrances.length && i < 2; i++) {
        const cid = roadNet.getRoadClusterId(entrances[i].gx, entrances[i].gy);
        if (cid > 0) entranceClusters.add(cid);
      }
    }

    let sawLockedRoadUnreachable = false;
    const zoneLimit = Math.min(6, candidateZones.length);

    for (let zi = 0; zi < zoneLimit; zi++) {
      const { zone, availCount } = candidateZones[zi];
      const stopLimit = Math.min(6, zone.roadStops.length);
      for (let si = 0; si < stopLimit; si++) {
        const stop = zone.roadStops[si];
        // O(1) cluster reachability pre-check BEFORE touching A*!
        if (roadNet && entranceClusters.size > 0) {
          const stopCid = roadNet.getRoadClusterId(stop.gx, stop.gy);
          if (stopCid > 0 && !entranceClusters.has(stopCid)) {
            sawLockedRoadUnreachable = true;
            continue;
          }
        }

        const entLimit = Math.min(2, entrances.length);
        for (let ei = 0; ei < entLimit; ei++) {
          const ent = entrances[ei];
          if (ent.gx === stop.gx && ent.gy === stop.gy) {
            return {
              stopRoad: { gx: stop.gx, gy: stop.gy },
              entrance: ent,
              path: [],
              zoneKey: zone.zoneKey,
              zoneCenterGx: zone.zGx,
              zoneCenterGy: zone.zGy,
              matureCount: availCount,
            };
          }
          const path = PathfindingSystem.findPath(
            ent.gx,
            ent.gy,
            stop.gx,
            stop.gy,
            chunks,
            'vehicle',
            680,
            { vehicleClass: 'heavy_truck', congestionMap }
          );
          if (path && path.length > 0) {
            const suit = PathfindingSystem.inspectRouteSuitability(path, chunks, 'heavy_truck');
            if (!suit.suitable && suit.sampleRestrictedCoord) {
              this.routeDiagnosticsMap.set(`heavy_local_forestry_${factory.id}`, {
                id: `heavy_local_forestry_${factory.id}`,
                category: 'vehicle_route',
                severity: 'warning',
                title: 'Logging Truck Using Narrow / Residential Road',
                detail: `Heavy logging route crossed ${suit.restrictedTiles} narrow/residential tile(s). Upgrade to Two-Lane or Industrial Road.`,
                gx: suit.sampleRestrictedCoord.gx,
                gy: suit.sampleRestrictedCoord.gy,
              });
            } else {
              this.routeDiagnosticsMap.delete(`heavy_local_forestry_${factory.id}`);
            }

            this.factoryDispatchBackoff.delete(factory.id);
            return {
              stopRoad: { gx: stop.gx, gy: stop.gy },
              entrance: ent,
              path,
              zoneKey: zone.zoneKey,
              zoneCenterGx: zone.zGx,
              zoneCenterGy: zone.zGy,
              matureCount: availCount,
            };
          }
        }
      }
      sawLockedRoadUnreachable = true;
    }

    const failOutcome: ForestryDispatchOutcome = {
      reason: sawLockedRoadUnreachable
        ? 'Blocked: Place a Road Connector linking owned roads to locked-area forest roads'
        : 'Waiting for available locked-land forest trees',
      diagnostic: sawLockedRoadUnreachable
        ? {
            id: `forestry_connector_${factory.id}`,
            category: 'forestry',
            severity: 'critical',
            title: 'No Road Connector to Locked Forest',
            detail: 'Logging trucks require a Road Connector linking owned roads to the locked-area road network.',
            gx: cx,
            gy: cy,
          }
        : undefined,
    };
    if (!originRoad) {
      this.factoryDispatchBackoff.set(factory.id, { netVer, untilSimSec: simTimeSec + 2.2, outcome: failOutcome });
    }
    return failOutcome;
  }

  private pickNearestWorkerTree(
    truckGx: number,
    truckGy: number,
    chunks: ChunkManager,
    reservedTrees: Set<number>,
    maxRadius = 9
  ): { gx: number; gy: number } | null {
    const cx = Math.round(truckGx);
    const cy = Math.round(truckGy);
    let best: { gx: number; gy: number; score: number } | null = null;

    for (let dy = -maxRadius; dy <= maxRadius; dy++) {
      for (let dx = -maxRadius; dx <= maxRadius; dx++) {
        if (dx * dx + dy * dy > maxRadius * maxRadius) continue;
        const tx = cx + dx;
        const ty = cy + dy;
        const key = tileKey(tx, ty);
        if (reservedTrees.has(key)) continue;

        const chunk = chunks.getLoadedChunkForGrid(tx, ty);
        if (!chunk) continue;
        const index = localTileIndex(tx, ty);
        if ((chunk.flags[index] & TileFlag.Unlocked) !== 0) continue; // Locked land only
        const ft = chunk.featureType[index] as FeatureType;
        if (!isMatureTreeFeature(ft)) continue;

        const permitted = (chunk.flags[index] & TileFlag.LoggingPermit) !== 0;
        const score = Math.hypot(dx, dy) - (permitted ? 1.5 : 0);
        if (!best || score < best.score) {
          best = { gx: tx, gy: ty, score };
        }
      }
    }
    return best ? { gx: best.gx, gy: best.gy } : null;
  }

  private findRouteBetweenBuildings(
    fromGx: number,
    fromGy: number,
    destBuilding: BuildingInstance,
    chunks: ChunkManager,
    roadNet?: RoadNetworkSystem
  ): Array<{ gx: number; gy: number }> | null {
    const startX = Math.round(fromGx);
    const startY = Math.round(fromGy);
    if (!this.roadSet.has(tileKey(startX, startY))) return null;
    const fromCluster = roadNet ? roadNet.getRoadClusterId(startX, startY) : 0;
    const destEntrances = PathfindingSystem.findBuildingEntrances(destBuilding, chunks, 'vehicle');
    const congestionMap = roadNet?.getTileCongestionMap();
    for (let i = 0; i < destEntrances.length && i < 4; i++) {
      const ent = destEntrances[i];
      if (ent.gx === startX && ent.gy === startY) return [];
      if (roadNet && fromCluster > 0) {
        const entCluster = roadNet.getRoadClusterId(ent.gx, ent.gy);
        if (entCluster > 0 && entCluster !== fromCluster) continue;
      }
      const path = PathfindingSystem.findPath(
        startX,
        startY,
        ent.gx,
        ent.gy,
        chunks,
        'vehicle',
        680,
        { vehicleClass: 'heavy_truck', congestionMap }
      );
      if (path && path.length > 0) return path;
    }
    return null;
  }

  private scheduleRegrowthFallback(
    gx: number,
    gy: number,
    ft: FeatureType,
    chunks: ChunkManager,
    state: WorldEngineState,
    forestry?: ForestrySystem
  ): void {
    if (chunks.isUnlocked(gx, gy)) return;
    if (forestry) {
      forestry.scheduleLockedTreeRegrowth(gx, gy, ft, chunks, state);
      return;
    }
    const species = TREE_SPECIES[ft];
    if (!species || species.woodYield <= 0 || ft === FeatureType.Sapling || ft === FeatureType.MediumTree) return;
    state.treeRegrowthQueue ??= [];
    const idx = state.treeRegrowthQueue.findIndex((e) => e.gx === gx && e.gy === gy);
    const saplingAt = state.tick + 25;
    const mediumAt = saplingAt + 40;
    const entry = {
      gx,
      gy,
      species: ft,
      saplingAtTick: saplingAt,
      mediumAtTick: mediumAt,
      readyAtTick: mediumAt + 50,
      stage: 'cooldown' as const,
    };
    if (idx >= 0) state.treeRegrowthQueue[idx] = entry;
    else state.treeRegrowthQueue.push(entry);
  }

  private stepLoggingTrucks(
    dtSec: number,
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet: RoadNetworkSystem | undefined,
    simTimeSec: number,
    onEvent?: (ev: TransportDeliveryEvent) => void,
    forestry?: ForestrySystem
  ): void {
    if (this.loggingTrucks.length === 0) return;

    const buildingById = new Map<number, BuildingInstance>();
    const sawmills: BuildingInstance[] = [];
    for (let i = 0; i < state.buildings.length; i++) {
      const b = state.buildings[i];
      buildingById.set(b.id, b);
      if (b.status !== 'constructing' && b.enabled && SAWMILL_IDS.has(b.defId)) {
        sawmills.push(b);
      }
    }

    // Track ALL trees reserved by ANY worker across ALL logging trucks + zone congestion
    const reservedTrees = new Set<number>();
    const activeZoneCounts = new Map<number, number>();
    const occupiedVehicleTiles = new Map<number, number>();
    const factoryEntrancesCache = new Map<number, Array<{ gx: number; gy: number }>>();

    for (let i = 0; i < this.roadVehicles.length; i++) {
      const rv = this.roadVehicles[i];
      const k = tileKey(Math.round(rv.gx), Math.round(rv.gy));
      occupiedVehicleTiles.set(k, (occupiedVehicleTiles.get(k) ?? 0) + 1);
    }
    for (let i = 0; i < this.loggingTrucks.length; i++) {
      const t = this.loggingTrucks[i];
      if (t.onRoad) {
        const k = tileKey(Math.round(t.gx), Math.round(t.gy));
        occupiedVehicleTiles.set(k, (occupiedVehicleTiles.get(k) ?? 0) + 1);
      }
      if (t.reservedTreeGx !== undefined && t.reservedTreeGy !== undefined) {
        reservedTrees.add(tileKey(t.reservedTreeGx, t.reservedTreeGy));
      }
      if (t.workers) {
        for (let wi = 0; wi < t.workers.length; wi++) {
          const w = t.workers[wi];
          if (w.targetTreeGx !== null && w.targetTreeGy !== null) {
            reservedTrees.add(tileKey(w.targetTreeGx, w.targetTreeGy));
          }
        }
      }
      if (
        (t.state === 'to_forest' || t.state === 'worker_chopping') &&
        t.zoneCenterGx !== undefined &&
        t.zoneCenterGy !== undefined
      ) {
        const zk = tileKey(Math.floor(t.zoneCenterGx / 8), Math.floor(t.zoneCenterGy / 8));
        activeZoneCounts.set(zk, (activeZoneCounts.get(zk) ?? 0) + 1);
      }
    }

    const woodCap = Math.max(1, state.capacities.wood ?? 500);
    const woodFull = (state.resources.wood ?? 0) >= woodCap * 0.98;
    let forestryDispatchBudget = 2;

    for (let ti = 0; ti < this.loggingTrucks.length; ti++) {
      const truck = this.loggingTrucks[ti];
      if (truck.state === 'chopping') {
        truck.state = 'worker_chopping';
      }

      const factory = buildingById.get(truck.factoryId);
      if (!factory || !factory.enabled) continue;
      const def = BUILDINGS[factory.defId];
      if (!def) continue;
      const fp = getBuildingFootprint(def, factory.rotation ?? 0);
      const homeGx = factory.gx + fp.width * 0.5;
      const homeGy = factory.gy + fp.height * 0.5;
      const cargoCapacity = Math.min(40, Math.max(30, truck.cargoCapacity ?? 32 + (factory.level - 1) * 4));
      truck.cargoCapacity = cargoCapacity;
      const workerRatio = Math.max(0.25, Math.min(1.2, (factory.assignedWorkers || 1) / Math.max(1, def.maxWorkers)));
      const baseSpeed = 2.5 * (0.7 + workerRatio * 0.55);

      // 1. Disembarked 5-6 Forestry Workers harvesting loop while truck stays parked on locked road
      if (truck.state === 'worker_chopping') {
        truck.onRoad = this.roadSet.has(tileKey(Math.round(truck.gx), Math.round(truck.gy)));
        truck.vx = 0;
        truck.vy = 0;
        truck.speed = 0;
        truck.workStatus = 'loading';

        if (!truck.workers || truck.workers.length === 0) {
          const workerCount = 5 + (truck.id % 2); // 5 to 6 workers disembark
          truck.workers = [];
          for (let wi = 0; wi < workerCount; wi++) {
            const angle = (wi / workerCount) * Math.PI * 2;
            const offsetGx = truck.gx + Math.cos(angle) * 0.35;
            const offsetGy = truck.gy + Math.sin(angle) * 0.35;
            const tree = this.pickNearestWorkerTree(truck.gx, truck.gy, chunks, reservedTrees, 9);
            if (tree) {
              reservedTrees.add(tileKey(tree.gx, tree.gy));
            }
            truck.workers.push({
              id: this.nextWorkerId++,
              gx: offsetGx,
              gy: offsetGy,
              vx: 0,
              vy: 0,
              phase: tree ? 'to_tree' : 'boarding',
              timer: 0.15 + wi * 0.1,
              targetTreeGx: tree ? tree.gx : null,
              targetTreeGy: tree ? tree.gy : null,
              carryingTimber: 0,
              facing: offsetGx - offsetGy >= 0 ? 1 : -1,
            });
          }
        }

        let allBoarded = true;
        for (let wi = 0; wi < truck.workers.length; wi++) {
          const w = truck.workers[wi];
          if (w.phase === 'exiting') {
            allBoarded = false;
            const tree =
              truck.cargoTimber < cargoCapacity
                ? this.pickNearestWorkerTree(truck.gx, truck.gy, chunks, reservedTrees, 9)
                : null;
            if (tree) {
              reservedTrees.add(tileKey(tree.gx, tree.gy));
              w.targetTreeGx = tree.gx;
              w.targetTreeGy = tree.gy;
              w.phase = 'to_tree';
            } else {
              w.phase = 'boarding';
            }
            continue;
          }

          if (w.phase === 'to_tree') {
            allBoarded = false;
            if (w.targetTreeGx === null || w.targetTreeGy === null) {
              w.phase = 'boarding';
              continue;
            }
            const dx = w.targetTreeGx - w.gx;
            const dy = w.targetTreeGy - w.gy;
            const dist = Math.hypot(dx, dy);
            if (dist > 0.16) {
              const step = Math.min(dist, 2.15 * dtSec);
              w.vx = dx / dist;
              w.vy = dy / dist;
              w.gx += w.vx * step;
              w.gy += w.vy * step;
              w.facing = w.vx - w.vy >= 0 ? 1 : -1;
            } else {
              w.gx = w.targetTreeGx;
              w.gy = w.targetTreeGy;
              w.vx = 0;
              w.vy = 0;
              w.phase = 'chopping';
              w.timer = 1.05;
            }
            continue;
          }

          if (w.phase === 'chopping') {
            allBoarded = false;
            w.timer -= dtSec;
            if (w.timer > 0) continue;

            const tx = Math.round(w.targetTreeGx ?? truck.gx);
            const ty = Math.round(w.targetTreeGy ?? truck.gy);
            const { cx, cy, index } = gridToChunk(tx, ty);
            const chunk = chunks.getLoadedChunk(cx, cy) ?? chunks.getOrCreateChunk(cx, cy);
            const ft = chunk.featureType[index] as FeatureType;
            const unlocked = (chunk.flags[index] & TileFlag.Unlocked) !== 0;

            if (!unlocked && isMatureTreeFeature(ft)) {
              const species = TREE_SPECIES[ft];
              chunk.featureType[index] = FeatureType.None;
              chunk.featureHp[index] = 0;
              chunks.markChunkModified(cx, cy);
              this.scheduleRegrowthFallback(tx, ty, ft, chunks, state, forestry);
              this.forestChopsSinceIndex++;
              if (this.forestChopsSinceIndex >= 8) {
                this.forestIndexDirty = true;
              }

              const lvlMult = getBuildingLevelMultiplier(factory.level, def);
              const woodAmt = Math.max(2, Math.round((species?.woodYield ?? 5) * lvlMult));
              w.carryingTimber = woodAmt;
            } else {
              w.carryingTimber = 0;
            }

            reservedTrees.delete(tileKey(tx, ty));
            w.targetTreeGx = null;
            w.targetTreeGy = null;
            w.phase = 'carrying_back';
            continue;
          }

          if (w.phase === 'carrying_back') {
            allBoarded = false;
            const dx = truck.gx - w.gx;
            const dy = truck.gy - w.gy;
            const dist = Math.hypot(dx, dy);
            if (dist > 0.16) {
              const step = Math.min(dist, 1.85 * dtSec);
              w.vx = dx / dist;
              w.vy = dy / dist;
              w.gx += w.vx * step;
              w.gy += w.vy * step;
              w.facing = w.vx - w.vy >= 0 ? 1 : -1;
              continue;
            }

            w.gx = truck.gx;
            w.gy = truck.gy;
            w.vx = 0;
            w.vy = 0;

            if (w.carryingTimber > 0 && truck.cargoTimber < cargoCapacity) {
              truck.cargoTimber = Math.min(cargoCapacity, truck.cargoTimber + 1);
              truck.treesHarvestedTrip = (truck.treesHarvestedTrip ?? 0) + 1;
              this.pendingWoodByTruckId.set(
                truck.id,
                (this.pendingWoodByTruckId.get(truck.id) ?? 0) + w.carryingTimber
              );
              state.stats.treesCut += 1;
              state.loggingStats ??= {
                harvestedTrees: 0,
                factoryTimberStock: 0,
                dailyTimberProduction: 0,
                activeTrucks: this.loggingTrucks.length,
                permittedForestTiles: 0,
              };
              state.loggingStats.harvestedTrees += 1;
            }
            w.carryingTimber = 0;

            if (truck.cargoTimber < cargoCapacity) {
              const nextTree = this.pickNearestWorkerTree(truck.gx, truck.gy, chunks, reservedTrees, 9);
              if (nextTree) {
                reservedTrees.add(tileKey(nextTree.gx, nextTree.gy));
                w.targetTreeGx = nextTree.gx;
                w.targetTreeGy = nextTree.gy;
                w.phase = 'to_tree';
                continue;
              }
            }
            w.phase = 'boarding';
            continue;
          }

          // boarding phase -> move smoothly back into truck cab
          const dx = truck.gx - w.gx;
          const dy = truck.gy - w.gy;
          const dist = Math.hypot(dx, dy);
          if (dist > 0.14) {
            allBoarded = false;
            const step = Math.min(dist, 2.0 * dtSec);
            w.gx += (dx / dist) * step;
            w.gy += (dy / dist) * step;
          } else {
            w.gx = truck.gx;
            w.gy = truck.gy;
          }
        }

        // Sync first active worker onto legacy fields for compatibility
        const lead = truck.workers[0];
        if (lead) {
          truck.workerGx = lead.gx;
          truck.workerGy = lead.gy;
          truck.workerFacing = lead.facing;
          truck.workerPhase =
            lead.phase === 'chopping'
              ? 'chopping'
              : lead.phase === 'carrying_back'
                ? 'returning_to_truck'
                : 'walking_to_tree';
        }

        truck.statusReason = `Crew (${truck.workers.length}) loading: ${truck.cargoTimber}/${cargoCapacity} trees`;

        if (!allBoarded) continue;

        // Crew has boarded! Clear workers array.
        truck.workers = [];
        truck.workerPhase = undefined;
        truck.workerGx = undefined;
        truck.workerGy = undefined;

        // If zone depleted & truck not full yet, relocate to next mature locked forest zone before returning!
        if (truck.cargoTimber < Math.min(cargoCapacity, 30)) {
          if (truck.zoneCenterGx !== undefined && truck.zoneCenterGy !== undefined) {
            const zk = tileKey(Math.floor(truck.zoneCenterGx / 8), Math.floor(truck.zoneCenterGy / 8));
            this.depletedZones.set(zk, state.tick + 90);
          }
          const relocate = this.findForestryWorkZoneAndRoute(
            factory,
            { gx: Math.round(truck.gx), gy: Math.round(truck.gy) },
            state,
            chunks,
            reservedTrees,
            activeZoneCounts,
            roadNet,
            simTimeSec
          );
          if (relocate.stopRoad && relocate.path && relocate.path.length > 0) {
            truck.stopRoadGx = relocate.stopRoad.gx;
            truck.stopRoadGy = relocate.stopRoad.gy;
            truck.targetGx = relocate.stopRoad.gx;
            truck.targetGy = relocate.stopRoad.gy;
            truck.zoneCenterGx = relocate.zoneCenterGx;
            truck.zoneCenterGy = relocate.zoneCenterGy;
            truck.assignedDestination = `Zone (${relocate.zoneCenterGx},${relocate.zoneCenterGy})`;
            truck.route = relocate.path;
            truck.routeIdx = 0;
            truck.routeVersion = roadNet?.networkVersion ?? 1;
            truck.prevGx = Math.round(truck.gx);
            truck.prevGy = Math.round(truck.gy);
            truck.nextGx = relocate.path[0].gx;
            truck.nextGy = relocate.path[0].gy;
            truck.progress = 0;
            truck.state = 'to_forest';
            truck.workStatus = 'travelling';
            truck.statusReason = `Relocating to next forest stand (${truck.cargoTimber}/${cargoCapacity})`;
            continue;
          }
        }

        // Return to factory along connected roads
        const backRoute = this.findRouteBetweenBuildings(truck.gx, truck.gy, factory, chunks, roadNet);
        if (backRoute) {
          truck.state = 'to_factory';
          truck.workStatus = 'returning';
          truck.targetGx = homeGx;
          truck.targetGy = homeGy;
          truck.assignedDestination = `${def.name} #${factory.id}`;
          truck.route = backRoute;
          truck.routeIdx = 0;
          truck.routeVersion = roadNet?.networkVersion ?? 1;
          truck.prevGx = Math.round(truck.gx);
          truck.prevGy = Math.round(truck.gy);
          truck.nextGx = backRoute[0]?.gx ?? Math.round(truck.gx);
          truck.nextGy = backRoute[0]?.gy ?? Math.round(truck.gy);
          truck.progress = backRoute.length > 0 ? 0 : 1;
          truck.statusReason = `Returning ${truck.cargoTimber}/${cargoCapacity} trees to factory`;
        } else {
          truck.state = 'blocked';
          truck.workStatus = 'waiting';
          truck.route = undefined;
          truck.harvestTimer = 1.8;
          truck.statusReason = 'Blocked: Road or Connector back to factory missing';
        }
        continue;
      }

      // 2. Idle or blocked -> schedule intelligent dispatch or resume return
      if (truck.state === 'idle_at_factory' || truck.state === 'blocked') {
        truck.harvestTimer -= dtSec;
        if (truck.harvestTimer > 0) continue;

        if (truck.cargoTimber > 0 && this.roadSet.has(tileKey(Math.round(truck.gx), Math.round(truck.gy)))) {
          const recovery = this.findRouteBetweenBuildings(truck.gx, truck.gy, factory, chunks, roadNet);
          if (recovery) {
            truck.state = 'to_factory';
            truck.workStatus = 'returning';
            truck.route = recovery;
            truck.routeIdx = 0;
            truck.routeVersion = roadNet?.networkVersion ?? 1;
            truck.prevGx = Math.round(truck.gx);
            truck.prevGy = Math.round(truck.gy);
            truck.nextGx = recovery[0]?.gx ?? Math.round(truck.gx);
            truck.nextGy = recovery[0]?.gy ?? Math.round(truck.gy);
            truck.progress = recovery.length > 0 ? 0 : 1;
            truck.statusReason = 'Resumed road route to factory';
            continue;
          }
        }

        if (woodFull) {
          truck.state = 'idle_at_factory';
          truck.workStatus = 'waiting';
          truck.harvestTimer = 2.0 + (truck.id % 4) * 0.2;
          truck.assignedDestination = undefined;
          truck.statusReason = 'Waiting: Timber warehouse storage full';
          continue;
        }

        if (factory.assignedWorkers <= 0) {
          truck.state = 'idle_at_factory';
          truck.workStatus = 'waiting';
          truck.harvestTimer = 2.0 + (truck.id % 4) * 0.2;
          truck.assignedDestination = undefined;
          truck.statusReason = 'Waiting: Assign factory workers';
          continue;
        }

        // Check if dispatch backoff is cached for this factory before consuming A* budget
        const backoff = this.factoryDispatchBackoff.get(factory.id);
        const netVer = roadNet ? roadNet.networkVersion : -2;
        if (backoff && backoff.netVer === netVer && simTimeSec < backoff.untilSimSec) {
          const cachedFail = backoff.outcome;
          if (cachedFail.diagnostic) {
            this.routeDiagnosticsMap.set(cachedFail.diagnostic.id, cachedFail.diagnostic);
          }
          truck.state = 'blocked';
          truck.workStatus = 'waiting';
          truck.harvestTimer = 2.2 + (truck.id % 5) * 0.22;
          truck.assignedDestination = undefined;
          truck.statusReason = cachedFail.reason ?? 'Waiting: No valid road route';
          continue;
        }

        if (forestryDispatchBudget <= 0) {
          // Stagger cleanly to next frame so multiple trucks never spike a single frame
          truck.harvestTimer = 0.2 + (truck.id % 4) * 0.08;
          continue;
        }
        forestryDispatchBudget--;

        let entrances = factoryEntrancesCache.get(factory.id);
        if (!entrances) {
          entrances = PathfindingSystem.findBuildingEntrances(factory, chunks, 'vehicle');
          factoryEntrancesCache.set(factory.id, entrances);
        }

        const plan = this.findForestryWorkZoneAndRoute(
          factory,
          undefined,
          state,
          chunks,
          reservedTrees,
          activeZoneCounts,
          roadNet,
          simTimeSec,
          entrances
        );

        if (plan.diagnostic) {
          this.routeDiagnosticsMap.set(plan.diagnostic.id, plan.diagnostic);
        } else {
          this.routeDiagnosticsMap.delete(`forestry_noroad_${factory.id}`);
          this.routeDiagnosticsMap.delete(`forestry_nocolroad_${factory.id}`);
          this.routeDiagnosticsMap.delete(`forestry_connector_${factory.id}`);
        }

        if (plan.stopRoad && plan.entrance && plan.path) {
          truck.stopRoadGx = plan.stopRoad.gx;
          truck.stopRoadGy = plan.stopRoad.gy;
          truck.targetGx = plan.stopRoad.gx;
          truck.targetGy = plan.stopRoad.gy;
          truck.zoneCenterGx = plan.zoneCenterGx;
          truck.zoneCenterGy = plan.zoneCenterGy;
          truck.assignedDestination = `Forest Zone (${plan.zoneCenterGx},${plan.zoneCenterGy})`;
          truck.treesHarvestedTrip = 0;
          truck.gx = plan.entrance.gx;
          truck.gy = plan.entrance.gy;
          truck.prevGx = plan.entrance.gx;
          truck.prevGy = plan.entrance.gy;
          truck.nextGx = plan.path[0]?.gx ?? plan.entrance.gx;
          truck.nextGy = plan.path[0]?.gy ?? plan.entrance.gy;
          truck.progress = plan.path.length > 0 ? 0 : 1;
          truck.onRoad = true;
          truck.route = plan.path;
          truck.routeIdx = 0;
          truck.routeVersion = roadNet?.networkVersion ?? 1;
          truck.state = 'to_forest';
          truck.workStatus = 'travelling';
          truck.blockedSec = 0;
          truck.statusReason = `En route to forest zone (${plan.matureCount ?? 0} mature trees)`;
          if (plan.zoneKey !== undefined) {
            activeZoneCounts.set(plan.zoneKey, (activeZoneCounts.get(plan.zoneKey) ?? 0) + 1);
          }
        } else {
          truck.state = 'blocked';
          truck.workStatus = 'waiting';
          truck.harvestTimer = 2.4 + (truck.id % 5) * 0.22;
          truck.assignedDestination = undefined;
          truck.statusReason = plan.reason ?? 'Waiting: No valid road route';
        }
        continue;
      }

      // 3. Strictly road-following driving states with signals, congestion rerouting & safe following
      if (roadNet && truck.routeVersion !== roadNet.networkVersion) {
        if (truck.state === 'to_forest' && truck.stopRoadGx !== undefined && truck.stopRoadGy !== undefined) {
          const curX = Math.round(truck.gx);
          const curY = Math.round(truck.gy);
          const curCid = roadNet.getRoadClusterId(curX, curY);
          const stopCid = roadNet.getRoadClusterId(truck.stopRoadGx, truck.stopRoadGy);
          const reroute =
            this.roadSet.has(tileKey(curX, curY)) && (curCid === 0 || stopCid === 0 || curCid === stopCid)
              ? PathfindingSystem.findPath(curX, curY, truck.stopRoadGx, truck.stopRoadGy, chunks, 'vehicle', 680, {
                  vehicleClass: 'heavy_truck',
                  congestionMap: roadNet.getTileCongestionMap(),
                })
              : null;
          if (reroute && reroute.length > 0) {
            truck.route = reroute;
            truck.routeIdx = 0;
            truck.routeVersion = roadNet.networkVersion;
            truck.prevGx = curX;
            truck.prevGy = curY;
            truck.nextGx = reroute[0].gx;
            truck.nextGy = reroute[0].gy;
            truck.progress = 0;
          } else {
            truck.route = undefined;
            truck.state = 'blocked';
            truck.workStatus = 'waiting';
            truck.harvestTimer = 1.6;
            truck.statusReason = 'Blocked: Road or Connector disconnected';
            continue;
          }
        } else if (truck.state === 'to_factory') {
          const back = this.findRouteBetweenBuildings(truck.gx, truck.gy, factory, chunks, roadNet);
          if (back) {
            truck.route = back;
            truck.routeIdx = 0;
            truck.routeVersion = roadNet.networkVersion;
          } else {
            truck.route = undefined;
            truck.state = 'blocked';
            truck.workStatus = 'waiting';
            truck.harvestTimer = 1.6;
            truck.statusReason = 'Blocked: Road back to factory disconnected';
            continue;
          }
        }
      }

      if (!truck.route) {
        truck.state = 'blocked';
        truck.workStatus = 'waiting';
        truck.harvestTimer = 1.5;
        truck.statusReason = 'Blocked: No road route available';
        continue;
      }

      const routeIdx = truck.routeIdx ?? 0;
      if (routeIdx < truck.route.length) {
        const curGx = Math.round(truck.gx);
        const curGy = Math.round(truck.gy);
        const progress = truck.progress ?? 1;

        if (progress >= 1 || (truck.nextGx === truck.prevGx && truck.nextGy === truck.prevGy)) {
          const wp = truck.route[routeIdx];
          const wpKey = tileKey(wp.gx, wp.gy);
          const dx = wp.gx - curGx;
          const dy = wp.gy - curGy;
          const manhattan = Math.abs(dx) + Math.abs(dy);

          const validEdge =
            this.roadSet.has(wpKey) &&
            (manhattan === 0 ||
              (manhattan === 1 && PathfindingSystem.isEdgeTraversable(curGx, curGy, wp.gx, wp.gy, chunks, 'vehicle')));

          if (!validEdge) {
            truck.route = undefined;
            truck.routeIdx = 0;
            truck.state = 'blocked';
            truck.workStatus = 'waiting';
            truck.harvestTimer = 1.5;
            truck.statusReason = 'Blocked: Road path severed';
            continue;
          }

          if (roadNet && manhattan === 1) {
            const sig = roadNet.getSignalState(wp.gx, wp.gy, dx, dy, simTimeSec, chunks);
            if (sig === 'red') {
              truck.speed = 0;
              truck.waitTimer = (truck.waitTimer ?? 0) + dtSec;
              truck.workStatus = 'waiting';
              continue;
            }
          }

          const occ = occupiedVehicleTiles.get(wpKey) ?? 0;
          const wpChunk = chunks.getLoadedChunkForGrid(wp.gx, wp.gy);
          const nrt = wpChunk ? (wpChunk.roadType[localTileIndex(wp.gx, wp.gy)] as RoadType) : RoadType.None;
          const laneCap = Math.max(2, getRoadProperties(nrt).lanesPerDirection + 1);
          if (manhattan === 1 && occ >= laneCap) {
            truck.speed = 0;
            truck.blockedSec = (truck.blockedSec ?? 0) + dtSec;
            truck.workStatus = 'waiting';
            if ((truck.blockedSec ?? 0) > 3.5 && truck.stopRoadGx !== undefined && truck.stopRoadGy !== undefined) {
              if (forestryDispatchBudget > 0) {
                forestryDispatchBudget--;
                const goal =
                  truck.state === 'to_forest'
                    ? { gx: truck.stopRoadGx, gy: truck.stopRoadGy }
                    : truck.route[truck.route.length - 1];
                const alt = PathfindingSystem.findPath(curGx, curGy, goal.gx, goal.gy, chunks, 'vehicle', 480, {
                  vehicleClass: 'heavy_truck',
                  congestionMap: roadNet?.getTileCongestionMap(),
                });
                if (alt && alt.length > 0) {
                  truck.route = alt;
                  truck.routeIdx = 0;
                }
                truck.blockedSec = 0;
              }
            }
            continue;
          }

          truck.prevGx = curGx;
          truck.prevGy = curGy;
          truck.nextGx = wp.gx;
          truck.nextGy = wp.gy;
          truck.vx = dx;
          truck.vy = dy;
          if (manhattan === 1) {
            truck.facing = dx - dy >= 0 ? 1 : -1;
          }
          truck.progress = manhattan === 0 ? 1 : 0;
          truck.waitTimer = 0;
          truck.blockedSec = 0;
          truck.workStatus = truck.state === 'to_forest' ? 'travelling' : 'returning';
        }

        const pGx = truck.prevGx ?? curGx;
        const pGy = truck.prevGy ?? curGy;
        const nGx = truck.nextGx ?? pGx;
        const nGy = truck.nextGy ?? pGy;
        const stepDx = nGx - pGx;
        const stepDy = nGy - pGy;

        if (Math.abs(stepDx) + Math.abs(stepDy) === 1) {
          const curChunk = chunks.getLoadedChunkForGrid(curGx, curGy);
          const rt = curChunk ? (curChunk.roadType[localTileIndex(curGx, curGy)] as RoadType) : RoadType.None;
          const roadSpeedMult = ROADS[rt]?.speedMultiplier ?? 1.0;
          const cong = roadNet?.getTileCongestion(curGx, curGy) ?? 0;
          const desiredSpeed = baseSpeed * Math.min(2.1, roadSpeedMult) * Math.max(0.45, 1 - cong * 0.4);
          truck.speed = desiredSpeed;
          truck.progress = Math.min(1, (truck.progress ?? 0) + desiredSpeed * dtSec);
          truck.gx = pGx + stepDx * truck.progress;
          truck.gy = pGy + stepDy * truck.progress;
          truck.onRoad = this.roadSet.has(tileKey(curGx, curGy));
          truck.laneOffset = truck.onRoad ? getLaneOffsetForTile(curGx, curGy, truck.id, chunks) : 0;

          if (truck.progress < 1) continue;
        }

        truck.gx = nGx;
        truck.gy = nGy;
        truck.routeIdx = routeIdx + 1;
        if (truck.routeIdx < truck.route.length) {
          continue;
        }
      }

      // Arrived at route destination
      truck.vx = 0;
      truck.vy = 0;
      truck.speed = 0;

      if (truck.state === 'to_forest') {
        truck.onRoad = true;
        truck.state = 'worker_chopping';
        truck.workStatus = 'loading';
        truck.workers = [];
        truck.statusReason = '5-6 forestry workers deploying to harvest mature trees';
        continue;
      }

      if (truck.state === 'to_factory') {
        const pendingWood = this.pendingWoodByTruckId.get(truck.id) ?? truck.cargoTimber * 4;
        if (truck.cargoTimber > 0 && pendingWood > 0) {
          const added = ResourceSystem.addClamped(state, 'wood', pendingWood);
          factory.realizedOutputTotal = (factory.realizedOutputTotal ?? 0) + added;
          this.harvestedToday += added;
          state.loggingStats ??= {
            harvestedTrees: 0,
            factoryTimberStock: 0,
            dailyTimberProduction: 0,
            activeTrucks: this.loggingTrucks.length,
            permittedForestTiles: 0,
          };
          state.loggingStats.factoryTimberStock = Math.round(state.resources.wood ?? 0);
          state.loggingStats.dailyTimberProduction = this.harvestedToday;
          if (added > 0) {
            onEvent?.({
              gx: homeGx,
              gy: homeGy,
              label: `+${added} Timber (${truck.cargoTimber} trees)`,
              color: '#a3e635',
            });
          }
        }
        truck.cargoTimber = 0;
        truck.treesHarvestedTrip = 0;
        this.pendingWoodByTruckId.delete(truck.id);

        // Optional logistics dispatch of processed wood to a Sawmill
        if (sawmills.length > 0 && truck.id % 3 === 0 && (state.resources.wood ?? 0) >= 6) {
          const mill = sawmills[truck.id % sawmills.length];
          const millRoute = this.findRouteBetweenBuildings(truck.gx, truck.gy, mill, chunks, roadNet);
          if (millRoute && millRoute.length > 0) {
            truck.cargoTimber = 6;
            this.pendingWoodByTruckId.set(truck.id, 6);
            truck.state = 'delivering_sawmill';
            truck.workStatus = 'travelling';
            truck.assignedDestination = `${BUILDINGS[mill.defId]?.name ?? 'Sawmill'} #${mill.id}`;
            truck.route = millRoute;
            truck.routeIdx = 0;
            truck.routeVersion = roadNet?.networkVersion ?? 1;
            truck.prevGx = Math.round(truck.gx);
            truck.prevGy = Math.round(truck.gy);
            truck.nextGx = millRoute[0].gx;
            truck.nextGy = millRoute[0].gy;
            truck.progress = 0;
            truck.onRoad = true;
            truck.statusReason = 'Transporting timber to Sawmill';
            continue;
          }
        }

        truck.state = 'idle_at_factory';
        truck.workStatus = 'idle';
        truck.onRoad = false;
        truck.harvestTimer = 0.6 + (truck.id % 4) * 0.18;
        truck.assignedDestination = undefined;
        truck.route = undefined;
        truck.statusReason = 'Unloaded timber at factory';
        continue;
      }

      if (truck.state === 'delivering_sawmill') {
        truck.cargoTimber = 0;
        this.pendingWoodByTruckId.delete(truck.id);
        if (sawmills.length > 0) {
          const mill = sawmills[truck.id % sawmills.length];
          if (mill.status === 'active') {
            mill.prodProgress = Math.min(0.98, mill.prodProgress + 0.15);
          }
        }
        const backRoute = this.findRouteBetweenBuildings(truck.gx, truck.gy, factory, chunks, roadNet);
        if (backRoute) {
          truck.state = 'to_factory';
          truck.workStatus = 'returning';
          truck.assignedDestination = `${def.name} #${factory.id}`;
          truck.route = backRoute;
          truck.routeIdx = 0;
          truck.routeVersion = roadNet?.networkVersion ?? 1;
          truck.prevGx = Math.round(truck.gx);
          truck.prevGy = Math.round(truck.gy);
          truck.nextGx = backRoute[0]?.gx ?? Math.round(truck.gx);
          truck.nextGy = backRoute[0]?.gy ?? Math.round(truck.gy);
          truck.progress = backRoute.length > 0 ? 0 : 1;
          truck.statusReason = 'Returning empty from Sawmill';
        } else {
          truck.state = 'idle_at_factory';
          truck.workStatus = 'idle';
          truck.assignedDestination = undefined;
          truck.route = undefined;
          truck.harvestTimer = 1.0;
        }
      }
    }
  }

  private syncTrains(state: WorldEngineState): void {
    if (this.cachedRailTiles.length < 3) {
      this.trains.length = 0;
      return;
    }

    const stationCount = state.buildings.filter((b) => {
      const def = BUILDINGS[b.defId];
      return def?.transitKind === 'rail' && b.status !== 'constructing' && b.enabled;
    }).length;
    const hasFreightYard = state.buildings.some(
      (b) => b.defId === 'freight_rail_yard' && b.status !== 'constructing'
    );

    const desiredTrains = Math.min(
      8,
      Math.max(1, Math.min(Math.floor(this.cachedRailTiles.length / 6), 1 + stationCount))
    );

    while (this.trains.length < desiredTrains) {
      const start = this.cachedRailTiles[(this.trains.length * 5) % this.cachedRailTiles.length];
      this.trains.push({
        id: this.nextId++,
        kind: hasFreightYard && this.trains.length % 2 === 1 ? 'freight' : 'passenger',
        gx: start.gx,
        gy: start.gy,
        prevGx: start.gx,
        prevGy: start.gy,
        progress: 1,
        facing: 1,
        cars: 3,
      });
    }
    if (this.trains.length > desiredTrains) {
      this.trains.length = desiredTrains;
    }
  }

  private stepTrains(dtSec: number): void {
    if (this.trains.length === 0 || this.railSet.size === 0) return;

    for (let ti = 0; ti < this.trains.length; ti++) {
      const train = this.trains[ti];
      const tx = Math.round(train.gx);
      const ty = Math.round(train.gy);
      if (!this.railSet.has(tileKey(tx, ty))) {
        if (this.cachedRailTiles.length > 0) {
          const fallback = this.cachedRailTiles[train.id % this.cachedRailTiles.length];
          train.gx = fallback.gx;
          train.gy = fallback.gy;
          train.prevGx = fallback.gx;
          train.prevGy = fallback.gy;
          train.progress = 1;
        }
        continue;
      }

      if (train.progress >= 1) {
        const neighbors: Array<[number, number]> = [];
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ]) {
          const nx = tx + dx;
          const ny = ty + dy;
          if (this.railSet.has(tileKey(nx, ny))) {
            neighbors.push([nx, ny]);
          }
        }
        if (neighbors.length === 0) continue;

        const forward = neighbors.filter(([nx, ny]) => nx !== train.prevGx || ny !== train.prevGy);
        const pool = forward.length > 0 ? forward : neighbors;
        const pick = pool[(train.id + tx * 3 + ty * 7) % pool.length];
        train.prevGx = tx;
        train.prevGy = ty;
        train.progress = 0;
        const dx = pick[0] - tx;
        const dy = pick[1] - ty;
        train.facing = dx - dy >= 0 ? 1 : -1;
        train.gx = tx + dx * 0.01;
        train.gy = ty + dy * 0.01;
      } else {
        const dirX = Math.sign(train.gx - train.prevGx);
        const dirY = Math.sign(train.gy - train.prevGy);
        const destX = train.prevGx + dirX;
        const destY = train.prevGy + dirY;
        const speed = train.kind === 'passenger' ? 3.2 : 2.3;
        train.progress = Math.min(1, train.progress + speed * dtSec);
        train.gx = train.prevGx + dirX * train.progress;
        train.gy = train.prevGy + dirY * train.progress;
        if (train.progress >= 1) {
          train.gx = destX;
          train.gy = destY;
        }
      }
    }
  }

  private getRoadAccessibleBuildings(
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet?: RoadNetworkSystem
  ): BuildingInstance[] {
    const out: BuildingInstance[] = [];
    for (const b of state.buildings) {
      if (!b.enabled) continue;
      if (roadNet) {
        const access = roadNet.getCachedRoadAccess(b, chunks);
        if (access.accessPoints.some((ap) => ap.roadType !== RoadType.Rail)) {
          out.push(b);
        }
      } else if (PathfindingSystem.findBuildingEntrances(b, chunks, 'vehicle').length > 0) {
        out.push(b);
      }
    }
    return out;
  }

  private syncRoadVehicles(
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet?: RoadNetworkSystem
  ): void {
    if (this.cachedRoadTiles.length < 2) {
      this.roadVehicles.length = 0;
      return;
    }

    const roadBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);

    let busWanted = 0;
    let taxiWanted = 0;
    let carWanted = 0;
    let deliveryWanted = 0;
    let fishVanWanted = 0;
    let cargoWanted = 0;
    let containerWanted = 0;
    let tractorWanted = 0;
    let constructionWanted = 0;
    let emergencyWanted = 0;

    for (const b of roadBuildings) {
      const def = BUILDINGS[b.defId];
      if (!def) continue;

      if (b.status === 'constructing') {
        constructionWanted = Math.min(3, constructionWanted + 1);
        continue;
      }

      if (def.transitKind === 'bus') busWanted += 2;
      if (def.transitKind === 'taxi') taxiWanted += 2;
      if (def.category === 'residential') carWanted += 1;
      if (def.category === 'commercial' || def.subCategory === 'market') deliveryWanted += 1;
      if (def.subCategory === 'fishing' || b.defId === 'fish_market') fishVanWanted += 1;
      if (def.subCategory === 'farming' || b.defId === 'farm_equipment_depot') tractorWanted += 1;
      if (
        def.category === 'production' ||
        def.category === 'industrial' ||
        def.category === 'storage' ||
        def.subCategory === 'mega_factory'
      ) {
        cargoWanted += 1;
      }
      if (
        def.subCategory === 'trade_port' ||
        def.exportsSurplus ||
        def.importsUnavailable ||
        def.isCustoms ||
        def.transitKind === 'sea' ||
        def.transitKind === 'air'
      ) {
        containerWanted += 1;
      }
      if (def.category === 'healthcare' || (def.safetyBonus ?? 0) > 0) {
        emergencyWanted += 1;
      }
    }

    if (roadBuildings.length >= 2 && carWanted === 0 && state.population.total > 0) {
      carWanted = 1;
    }

    const desiredKinds: RoadVehicleKind[] = [];
    const pushCapped = (kind: RoadVehicleKind, count: number, cap: number) => {
      const n = Math.min(cap, count);
      for (let i = 0; i < n; i++) desiredKinds.push(kind);
    };

    pushCapped('bus', busWanted, 6);
    pushCapped('taxi', taxiWanted, 5);
    pushCapped('car', Math.floor((carWanted + 1) / 2), 6);
    pushCapped('delivery_van', deliveryWanted, 5);
    pushCapped('fish_van', fishVanWanted, 3);
    pushCapped('cargo_truck', cargoWanted, 6);
    pushCapped('container_truck', containerWanted, 4);
    pushCapped('tractor', tractorWanted, 3);
    pushCapped('construction_truck', constructionWanted, 3);
    pushCapped('emergency', emergencyWanted, 3);

    desiredKinds.length = Math.min(desiredKinds.length, 32);

    if (desiredKinds.length === 0) {
      this.roadVehicles.length = 0;
      return;
    }

    while (this.roadVehicles.length < desiredKinds.length) {
      const idx = this.roadVehicles.length;
      const kind = desiredKinds[idx];
      const start = this.pickSpawnRoadTile(kind, roadBuildings, chunks, idx);
      const vClass = getVehicleClass(kind);
      const veh: RoadVehicleAgent = {
        id: this.nextId++,
        kind,
        vehicleClass: vClass,
        gx: start.gx,
        gy: start.gy,
        prevGx: start.gx,
        prevGy: start.gy,
        nextGx: start.gx,
        nextGy: start.gy,
        progress: 1,
        facing: 1,
        vx: 0,
        vy: 0,
        laneOffset: 0,
        speed: 0,
        maxSpeed: getBaseMaxSpeed(kind),
        state: 'idle',
        waitTimer: 0.2 + ((idx * 0.21) % 2.1),
        blockedSec: 0,
        route: [],
        routeIdx: 0,
        routeVersion: 0,
      };
      this.roadVehicles.push(veh);
    }

    if (this.roadVehicles.length > desiredKinds.length) {
      this.roadVehicles.length = desiredKinds.length;
    }

    let syncTripBudget = 2;
    for (let i = 0; i < this.roadVehicles.length; i++) {
      const veh = this.roadVehicles[i];
      if (veh.kind !== desiredKinds[i]) {
        veh.kind = desiredKinds[i];
        veh.vehicleClass = getVehicleClass(desiredKinds[i]);
        veh.maxSpeed = getBaseMaxSpeed(desiredKinds[i]);
      }
      if (veh.route.length === 0 && veh.state === 'idle' && veh.waitTimer <= 0 && syncTripBudget > 0) {
        syncTripBudget--;
        this.assignTrip(veh, state, chunks, roadNet, roadBuildings);
      }
    }
  }

  private pickSpawnRoadTile(
    kind: RoadVehicleKind,
    roadBuildings: BuildingInstance[],
    chunks: ChunkManager,
    seedIdx: number
  ): { gx: number; gy: number } {
    if (roadBuildings.length > 0) {
      const b = roadBuildings[seedIdx % roadBuildings.length];
      const entrances = PathfindingSystem.findBuildingEntrances(b, chunks, 'vehicle');
      if (entrances.length > 0) return entrances[0];
    }
    void kind;
    return this.cachedRoadTiles[(seedIdx * 7) % this.cachedRoadTiles.length];
  }

  private pickSourceAndDest(
    kind: RoadVehicleKind,
    state: WorldEngineState,
    roadBuildings: BuildingInstance[],
    roadNet?: RoadNetworkSystem,
    seed = 1
  ): {
    source?: BuildingInstance;
    dest?: BuildingInstance;
    cargoResource?: ResourceId;
    cargoAmount?: number;
  } {
    if (roadBuildings.length === 0) return {};

    const constructing = roadBuildings.filter((b) => b.status === 'constructing');
    const active = roadBuildings.filter((b) => b.status !== 'constructing');
    const byMatch = (pred: (b: BuildingInstance) => boolean) => active.filter(pred);

    let sources: BuildingInstance[] = [];
    let dests: BuildingInstance[] = [];
    let cargoResource: ResourceId | undefined;
    let cargoAmount = 0;

    switch (kind) {
      case 'construction_truck': {
        sources = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(
            def &&
              (def.category === 'storage' ||
                def.category === 'industrial' ||
                def.category === 'government' ||
                b.defId === 'sawmill_mega')
          );
        });
        if (sources.length === 0) sources = active;
        dests = constructing;
        cargoResource = 'bricks';
        cargoAmount = 2;
        break;
      }
      case 'container_truck': {
        sources = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(
            def &&
              (def.subCategory === 'trade_port' ||
                def.exportsSurplus ||
                def.importsUnavailable ||
                def.isCustoms ||
                def.transitKind === 'sea' ||
                def.transitKind === 'air')
          );
        });
        dests = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(
            def && (def.category === 'storage' || def.category === 'industrial' || def.category === 'production')
          );
        });
        cargoResource = 'tools';
        cargoAmount = 3;
        break;
      }
      case 'cargo_truck': {
        sources = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(
            def && (def.category === 'production' || def.category === 'industrial' || def.subCategory === 'mega_factory')
          );
        });
        dests = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(def && (def.category === 'storage' || def.category === 'commercial' || def.subCategory === 'market'));
        });
        break;
      }
      case 'delivery_van': {
        sources = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(def && (def.category === 'agricultural' || def.category === 'storage' || def.category === 'production'));
        });
        dests = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(def && (def.category === 'commercial' || def.subCategory === 'market' || def.category === 'residential'));
        });
        cargoResource = 'bread';
        cargoAmount = 2;
        break;
      }
      case 'fish_van': {
        sources = byMatch((b) => BUILDINGS[b.defId]?.subCategory === 'fishing');
        dests = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(def && (b.defId === 'fish_market' || def.category === 'commercial' || def.subCategory === 'market'));
        });
        cargoResource = 'fish';
        cargoAmount = 2;
        break;
      }
      case 'tractor': {
        sources = byMatch((b) => b.defId === 'farm_equipment_depot' || BUILDINGS[b.defId]?.subCategory === 'farming');
        dests = byMatch((b) => BUILDINGS[b.defId]?.subCategory === 'farming' || BUILDINGS[b.defId]?.category === 'agricultural');
        cargoResource = 'wheat';
        cargoAmount = 2;
        break;
      }
      case 'emergency': {
        sources = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(def && (def.category === 'healthcare' || (def.safetyBonus ?? 0) > 0));
        });
        dests = byMatch((b) => BUILDINGS[b.defId]?.category === 'residential' || BUILDINGS[b.defId]?.category === 'industrial');
        break;
      }
      case 'bus':
      case 'taxi': {
        sources = byMatch((b) => BUILDINGS[b.defId]?.transitKind === kind || BUILDINGS[b.defId]?.category === 'residential');
        dests = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(def && (def.category === 'commercial' || def.category === 'government' || def.category === 'educational' || def.category === 'recreational'));
        });
        break;
      }
      default: {
        sources = byMatch((b) => BUILDINGS[b.defId]?.category === 'residential');
        dests = byMatch((b) => {
          const def = BUILDINGS[b.defId];
          return Boolean(def && (def.category === 'commercial' || def.category === 'government' || def.category === 'industrial' || def.subCategory === 'corporate'));
        });
        break;
      }
    }

    if (sources.length === 0) sources = active.length > 0 ? active : roadBuildings;
    if (dests.length === 0) dests = active.length > 0 ? active : roadBuildings;

    const source = sources[seed % sources.length];
    const sourceCluster = source.connectedRoadCluster;

    // Prevent duplicate identical active cargo assignments on same source+dest if alternatives exist
    const sameClusterDests = dests.filter(
      (d) => d.id !== source.id && (sourceCluster === 0 || d.connectedRoadCluster === sourceCluster)
    );
    const activePairs = new Set(
      this.roadVehicles
        .filter((v) => !v.returning && v.sourceBuildingId === source.id && v.destBuildingId)
        .map((v) => v.destBuildingId!)
    );
    const nonDupDests = sameClusterDests.filter((d) => !activePairs.has(d.id));
    const destPool = nonDupDests.length > 0 ? nonDupDests : sameClusterDests;

    const dest = destPool.length > 0 ? destPool[(seed * 3 + 1) % destPool.length] : undefined;

    if (!cargoResource && source) {
      const sDef = BUILDINGS[source.defId];
      const outKeys = sDef?.recipe ? (Object.keys(sDef.recipe.outputs) as ResourceId[]) : [];
      if (outKeys.length > 0) {
        cargoResource = outKeys[seed % outKeys.length];
        cargoAmount = 2;
      } else if (kind === 'cargo_truck') {
        cargoResource = CARGO_FALLBACK_RESOURCES[seed % CARGO_FALLBACK_RESOURCES.length];
        cargoAmount = 1;
      }
    }

    void state;
    void roadNet;
    return { source, dest, cargoResource, cargoAmount };
  }

  private assignTrip(
    veh: RoadVehicleAgent,
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet?: RoadNetworkSystem,
    cachedBuildings?: BuildingInstance[]
  ): boolean {
    const roadBuildings = cachedBuildings ?? this.getRoadAccessibleBuildings(state, chunks, roadNet);
    const curGx = Math.round(veh.gx);
    const curGy = Math.round(veh.gy);
    const onRoadNow = this.roadSet.has(tileKey(curGx, curGy));

    const { source, dest, cargoResource, cargoAmount } = this.pickSourceAndDest(
      veh.kind,
      state,
      roadBuildings,
      roadNet,
      veh.id + state.tick
    );

    veh.sourceBuildingId = source?.id;
    veh.destBuildingId = dest?.id;
    if (veh.returning) {
      veh.cargoResource = undefined;
      veh.cargoAmount = 0;
    } else {
      veh.cargoResource = cargoResource;
      veh.cargoAmount = cargoAmount;
    }

    const congestionMap = roadNet?.getTileCongestionMap();
    const pathOpts = { vehicleClass: veh.vehicleClass, congestionMap };

    const targetB = veh.returning ? source : dest ?? source;
    if (targetB) {
      const targetEntrances = PathfindingSystem.findBuildingEntrances(targetB, chunks, 'vehicle');
      let startTile = onRoadNow ? { gx: curGx, gy: curGy } : null;
      if (!startTile && source && veh.state === 'idle') {
        const srcEntrances = PathfindingSystem.findBuildingEntrances(source, chunks, 'vehicle');
        if (srcEntrances.length > 0) {
          startTile = srcEntrances[0];
          veh.gx = startTile.gx;
          veh.gy = startTile.gy;
          veh.prevGx = startTile.gx;
          veh.prevGy = startTile.gy;
        }
      }

      if (startTile && targetEntrances.length > 0) {
        const startCid = roadNet ? roadNet.getRoadClusterId(startTile.gx, startTile.gy) : 0;
        for (let i = 0; i < targetEntrances.length && i < 4; i++) {
          const endTile = targetEntrances[i];
          if (endTile.gx === startTile.gx && endTile.gy === startTile.gy) continue;
          if (roadNet && startCid > 0) {
            const endCid = roadNet.getRoadClusterId(endTile.gx, endTile.gy);
            if (endCid > 0 && endCid !== startCid) continue;
          }
          const path = PathfindingSystem.findPath(
            startTile.gx,
            startTile.gy,
            endTile.gx,
            endTile.gy,
            chunks,
            'vehicle',
            560,
            pathOpts
          );
          if (path && path.length > 0) {
            if (veh.vehicleClass === 'heavy_truck') {
              const suit = PathfindingSystem.inspectRouteSuitability(path, chunks, veh.vehicleClass);
              if (!suit.suitable && suit.sampleRestrictedCoord) {
                this.routeDiagnosticsMap.set(`heavy_local_freight_${veh.id}`, {
                  id: `heavy_local_freight_${veh.id}`,
                  category: 'vehicle_route',
                  severity: 'warning',
                  title: 'Freight Truck on Unsuitable Narrow Street',
                  detail: `Truck #${veh.id} (${veh.kind}) had to cross ${suit.restrictedTiles} narrow/residential tile(s) due to no arterial bypass.`,
                  gx: suit.sampleRestrictedCoord.gx,
                  gy: suit.sampleRestrictedCoord.gy,
                });
              } else {
                this.routeDiagnosticsMap.delete(`heavy_local_freight_${veh.id}`);
              }
            }

            veh.route = path;
            veh.routeIdx = 0;
            veh.routeVersion = roadNet?.networkVersion ?? 1;
            veh.prevGx = startTile.gx;
            veh.prevGy = startTile.gy;
            veh.nextGx = path[0].gx;
            veh.nextGy = path[0].gy;
            veh.progress = 0;
            veh.state = 'en_route';
            veh.blockedSec = 0;
            veh.statusReason = veh.returning ? 'Returning empty on road' : 'Delivering freight on road';
            return true;
          }
        }
      }
    }

    if (onRoadNow) {
      const clusterId = roadNet?.getRoadClusterId(curGx, curGy) ?? 0;
      const sameClusterTiles =
        clusterId > 0 && roadNet
          ? (this.cachedRoadTilesByCluster.get(clusterId) ?? this.cachedRoadTiles)
          : this.cachedRoadTiles;

      for (let attempt = 0; attempt < 5 && sameClusterTiles.length > 1; attempt++) {
        const pick = sameClusterTiles[(veh.id * 17 + state.tick * 7 + attempt * 13) % sameClusterTiles.length];
        if (pick.gx === curGx && pick.gy === curGy) continue;
        const path = PathfindingSystem.findPath(curGx, curGy, pick.gx, pick.gy, chunks, 'vehicle', 320, pathOpts);
        if (path && path.length > 0) {
          veh.route = path;
          veh.routeIdx = 0;
          veh.routeVersion = roadNet?.networkVersion ?? 1;
          veh.prevGx = curGx;
          veh.prevGy = curGy;
          veh.nextGx = path[0].gx;
          veh.nextGy = path[0].gy;
          veh.progress = 0;
          veh.state = 'en_route';
          veh.blockedSec = 0;
          veh.statusReason = 'Patrolling connected road network';
          return true;
        }
      }
    }

    veh.route = [];
    veh.routeIdx = 0;
    veh.state = 'idle';
    veh.speed = 0;
    veh.waitTimer = 1.6 + (veh.id % 5) * 0.25;
    veh.statusReason = 'Blocked: No connected road route';
    return false;
  }

  private rerouteOrRecover(
    veh: RoadVehicleAgent,
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet?: RoadNetworkSystem,
    cachedBuildings?: BuildingInstance[]
  ): void {
    const tx = Math.round(veh.gx);
    const ty = Math.round(veh.gy);
    let onRoadNow = this.roadSet.has(tileKey(tx, ty));
    if (!onRoadNow) {
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        const nx = tx + dx;
        const ny = ty + dy;
        if (this.roadSet.has(tileKey(nx, ny))) {
          veh.gx = nx;
          veh.gy = ny;
          veh.prevGx = nx;
          veh.prevGy = ny;
          veh.nextGx = nx;
          veh.nextGy = ny;
          veh.progress = 1;
          onRoadNow = true;
          break;
        }
      }
    }

    if (!onRoadNow) {
      veh.route = [];
      veh.routeIdx = 0;
      veh.state = 'idle';
      veh.speed = 0;
      veh.waitTimer = 2.0;
      veh.statusReason = 'Halted: Road removed underneath vehicle';
      return;
    }

    if (veh.route.length > 0) {
      const goal = veh.route[veh.route.length - 1];
      const curX = Math.round(veh.gx);
      const curY = Math.round(veh.gy);
      if (this.roadSet.has(tileKey(goal.gx, goal.gy))) {
        const curCid = roadNet ? roadNet.getRoadClusterId(curX, curY) : 0;
        const goalCid = roadNet ? roadNet.getRoadClusterId(goal.gx, goal.gy) : 0;
        if (curCid === 0 || goalCid === 0 || curCid === goalCid) {
          const newPath = PathfindingSystem.findPath(curX, curY, goal.gx, goal.gy, chunks, 'vehicle', 480, {
            vehicleClass: veh.vehicleClass,
            congestionMap: roadNet?.getTileCongestionMap(),
          });
          if (newPath && newPath.length > 0) {
            veh.route = newPath;
            veh.routeIdx = 0;
            veh.routeVersion = roadNet?.networkVersion ?? 1;
            veh.prevGx = curX;
            veh.prevGy = curY;
            veh.nextGx = newPath[0].gx;
            veh.nextGy = newPath[0].gy;
            veh.progress = 0;
            veh.state = 'en_route';
            veh.blockedSec = 0;
            veh.statusReason = 'Rerouted on road network';
            return;
          }
        }
      }
    }

    this.assignTrip(veh, state, chunks, roadNet, cachedBuildings);
  }

  private applyDeliveryCompletion(
    veh: RoadVehicleAgent,
    state: WorldEngineState,
    onEvent?: (ev: TransportDeliveryEvent) => void
  ): void {
    if (veh.returning) return;
    const dest = veh.destBuildingId ? state.buildings.find((b) => b.id === veh.destBuildingId) : undefined;
    if (!dest) return;

    if (veh.kind === 'construction_truck' && dest.status === 'constructing') {
      dest.buildProgress = Math.min(0.99, dest.buildProgress + 0.04);
      if (veh.id % 2 === 0) {
        onEvent?.({ gx: dest.gx, gy: dest.gy, label: '+4% Build Supply', color: '#fbbf24' });
      }
      return;
    }

    if (
      veh.cargoResource &&
      veh.cargoAmount &&
      veh.cargoAmount > 0 &&
      (veh.kind === 'cargo_truck' ||
        veh.kind === 'container_truck' ||
        veh.kind === 'delivery_van' ||
        veh.kind === 'fish_van' ||
        veh.kind === 'tractor')
    ) {
      const added = ResourceSystem.addClamped(state, veh.cargoResource, veh.cargoAmount);
      if (added > 0 && veh.id % 5 === 0) {
        onEvent?.({ gx: dest.gx, gy: dest.gy, label: `+${added} Freight`, color: '#38bdf8' });
      }
      const def = BUILDINGS[dest.defId];
      if (def?.exportsSurplus || def?.category === 'commercial') {
        state.coins += 1;
        state.stats.totalCoinsEarned += 1;
      }
    }
  }

  private stepRoadVehicles(
    dtSec: number,
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet: RoadNetworkSystem | undefined,
    simTimeSec: number,
    onEvent?: (ev: TransportDeliveryEvent) => void
  ): void {
    if (this.roadVehicles.length === 0 || this.roadSet.size === 0) return;

    const tileBuckets = new Map<number, RoadVehicleAgent[]>();
    for (let i = 0; i < this.roadVehicles.length; i++) {
      const v = this.roadVehicles[i];
      const key = tileKey(Math.round(v.gx), Math.round(v.gy));
      let list = tileBuckets.get(key);
      if (!list) {
        list = [];
        tileBuckets.set(key, list);
      }
      list.push(v);
    }

    const netVer = roadNet?.networkVersion ?? 1;
    let aStarBudget = 3;
    let cachedBuildings: BuildingInstance[] | undefined = undefined;

    for (let vi = 0; vi < this.roadVehicles.length; vi++) {
      const veh = this.roadVehicles[vi];
      const tx = Math.round(veh.gx);
      const ty = Math.round(veh.gy);

      if (!this.roadSet.has(tileKey(tx, ty))) {
        if (aStarBudget > 0) {
          aStarBudget--;
          if (!cachedBuildings) cachedBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);
          this.rerouteOrRecover(veh, state, chunks, roadNet, cachedBuildings);
        } else {
          veh.waitTimer = 0.2;
        }
        continue;
      }

      if (veh.state === 'loading' || veh.state === 'unloading' || veh.state === 'idle') {
        veh.speed = 0;
        veh.waitTimer -= dtSec;
        if (veh.waitTimer <= 0) {
          if (aStarBudget > 0) {
            aStarBudget--;
            if (veh.state === 'unloading') {
              veh.returning = !veh.returning;
            }
            if (!cachedBuildings) cachedBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);
            this.assignTrip(veh, state, chunks, roadNet, cachedBuildings);
          } else {
            veh.waitTimer = 0.18 + (veh.id % 4) * 0.06;
          }
        }
        continue;
      }

      if (roadNet && veh.routeVersion !== netVer) {
        let routeIntact = veh.route.length > 0 && veh.routeIdx < veh.route.length;
        if (routeIntact) {
          const lookaheadEnd = Math.min(veh.route.length, veh.routeIdx + 8);
          let px = tx;
          let py = ty;
          for (let i = veh.routeIdx; i < lookaheadEnd; i++) {
            const wp = veh.route[i];
            if (!PathfindingSystem.isEdgeTraversable(px, py, wp.gx, wp.gy, chunks, 'vehicle')) {
              routeIntact = false;
              break;
            }
            px = wp.gx;
            py = wp.gy;
          }
        }
        if (routeIntact) {
          veh.routeVersion = netVer;
        } else if (aStarBudget > 0) {
          aStarBudget--;
          if (!cachedBuildings) cachedBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);
          this.rerouteOrRecover(veh, state, chunks, roadNet, cachedBuildings);
          continue;
        } else {
          continue;
        }
      }

      if (veh.progress >= 1 || (veh.nextGx === veh.prevGx && veh.nextGy === veh.prevGy)) {
        if (veh.routeIdx >= veh.route.length) {
          if (!veh.returning) {
            this.applyDeliveryCompletion(veh, state, onEvent);
          }
          veh.cargoAmount = 0;
          veh.cargoResource = undefined;
          veh.state = 'unloading';
          veh.waitTimer = veh.returning ? 0.5 : 0.85;
          veh.speed = 0;
          continue;
        }

        const wp = veh.route[veh.routeIdx];
        const dx = wp.gx - tx;
        const dy = wp.gy - ty;

        if (Math.abs(dx) + Math.abs(dy) !== 1) {
          if (aStarBudget > 0) {
            aStarBudget--;
            if (!cachedBuildings) cachedBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);
            this.rerouteOrRecover(veh, state, chunks, roadNet, cachedBuildings);
          }
          continue;
        }

        if (!PathfindingSystem.isEdgeTraversable(tx, ty, wp.gx, wp.gy, chunks, 'vehicle')) {
          if (aStarBudget > 0) {
            aStarBudget--;
            if (!cachedBuildings) cachedBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);
            this.rerouteOrRecover(veh, state, chunks, roadNet, cachedBuildings);
          }
          continue;
        }

        if (roadNet && veh.vehicleClass !== 'emergency') {
          const sig = roadNet.getSignalState(wp.gx, wp.gy, dx, dy, simTimeSec, chunks);
          if (sig === 'red') {
            veh.state = 'waiting_signal';
            veh.speed = 0;
            veh.blockedSec += dtSec;
            continue;
          }
        }

        const nextOcc = tileBuckets.get(tileKey(wp.gx, wp.gy));
        if (nextOcc && nextOcc.length > 0) {
          const wpChunk = chunks.getLoadedChunkForGrid(wp.gx, wp.gy);
          const nrt = wpChunk ? (wpChunk.roadType[localTileIndex(wp.gx, wp.gy)] as RoadType) : RoadType.None;
          const cap = Math.max(1, getRoadProperties(nrt).lanesPerDirection) + 1;
          let blockedByAhead = nextOcc.length >= cap;
          if (!blockedByAhead) {
            for (let oi = 0; oi < nextOcc.length; oi++) {
              const other = nextOcc[oi];
              if (other.id !== veh.id && other.progress < 0.42 && (other.vx !== dx || other.vy !== dy)) {
                blockedByAhead = true;
                break;
              }
            }
          }
          if (blockedByAhead) {
            veh.state = 'waiting_traffic';
            veh.speed = 0;
            veh.blockedSec += dtSec;
            const staggerThresh = 3.5 + (veh.id % 7) * 0.35;
            if (veh.blockedSec > staggerThresh && aStarBudget > 0) {
              aStarBudget--;
              if (!cachedBuildings) cachedBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);
              this.rerouteOrRecover(veh, state, chunks, roadNet, cachedBuildings);
            }
            continue;
          }
        }

        veh.prevGx = tx;
        veh.prevGy = ty;
        veh.nextGx = wp.gx;
        veh.nextGy = wp.gy;
        veh.vx = dx;
        veh.vy = dy;
        veh.facing = dx - dy >= 0 ? 1 : -1;
        veh.progress = 0;
        veh.state = 'en_route';
        veh.blockedSec = 0;
      }

      const dirX = veh.nextGx - veh.prevGx;
      const dirY = veh.nextGy - veh.prevGy;
      if (Math.abs(dirX) + Math.abs(dirY) !== 1) {
        if (aStarBudget > 0) {
          aStarBudget--;
          if (!cachedBuildings) cachedBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);
          this.rerouteOrRecover(veh, state, chunks, roadNet, cachedBuildings);
        }
        continue;
      }

      let leaderCapSpeed = Infinity;
      const pKey = tileKey(veh.prevGx, veh.prevGy);
      const nKey = tileKey(veh.nextGx, veh.nextGy);
      const pList = tileBuckets.get(pKey);
      if (pList) {
        for (let pi = 0; pi < pList.length; pi++) {
          const peer = pList[pi];
          if (peer.id === veh.id) continue;
          if (peer.prevGx === veh.prevGx && peer.prevGy === veh.prevGy && peer.nextGx === veh.nextGx && peer.nextGy === veh.nextGy) {
            const gap = peer.progress - veh.progress;
            if (gap > 0 && gap < 0.44) {
              leaderCapSpeed = Math.min(leaderCapSpeed, peer.speed * 0.85);
              if (gap < 0.25) leaderCapSpeed = 0;
            }
          }
        }
      }
      if (nKey !== pKey) {
        const nList = tileBuckets.get(nKey);
        if (nList) {
          for (let pi = 0; pi < nList.length; pi++) {
            const peer = nList[pi];
            if (peer.id === veh.id) continue;
            if (peer.prevGx === veh.prevGx && peer.prevGy === veh.prevGy && peer.nextGx === veh.nextGx && peer.nextGy === veh.nextGy) {
              const gap = peer.progress - veh.progress;
              if (gap > 0 && gap < 0.44) {
                leaderCapSpeed = Math.min(leaderCapSpeed, peer.speed * 0.85);
                if (gap < 0.25) leaderCapSpeed = 0;
              }
            }
          }
        }
      }

      const curChunk = chunks.getLoadedChunkForGrid(tx, ty);
      const rt = curChunk ? (curChunk.roadType[localTileIndex(tx, ty)] as RoadType) : RoadType.None;
      const roadSpeedMult = ROADS[rt]?.speedMultiplier ?? 1.0;
      const cong = roadNet?.getTileCongestion(tx, ty) ?? 0;
      const desiredSpeed = Math.min(
        leaderCapSpeed,
        veh.maxSpeed * Math.min(2.3, roadSpeedMult) * Math.max(0.45, 1 - cong * 0.45)
      );

      if (desiredSpeed <= 0.05) {
        veh.speed = Math.max(0, veh.speed - 7.5 * dtSec);
        veh.state = 'waiting_traffic';
        veh.blockedSec += dtSec;
        const staggerThresh = 3.8 + (veh.id % 7) * 0.35;
        if (veh.blockedSec > staggerThresh && aStarBudget > 0) {
          aStarBudget--;
          if (!cachedBuildings) cachedBuildings = this.getRoadAccessibleBuildings(state, chunks, roadNet);
          this.rerouteOrRecover(veh, state, chunks, roadNet, cachedBuildings);
        }
        continue;
      }

      veh.state = 'en_route';
      veh.blockedSec = 0;
      veh.speed =
        veh.speed < desiredSpeed
          ? Math.min(desiredSpeed, veh.speed + 5.5 * dtSec)
          : Math.max(desiredSpeed, veh.speed - 7.0 * dtSec);

      veh.progress = Math.min(1, veh.progress + veh.speed * dtSec);
      veh.gx = veh.prevGx + dirX * veh.progress;
      veh.gy = veh.prevGy + dirY * veh.progress;
      veh.laneOffset = getLaneOffsetForTile(tx, ty, veh.id, chunks);

      if (veh.progress >= 1) {
        veh.gx = veh.nextGx;
        veh.gy = veh.nextGy;
        veh.routeIdx += 1;
      }
    }
  }
}
