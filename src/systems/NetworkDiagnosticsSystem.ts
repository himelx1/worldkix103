import type { ChunkManager } from '../core/ChunkManager';
import { WorldEngineState } from '../core/GameState';
import { unpackTileX, unpackTileY } from '../core/IsoMath';
import { BUILDINGS } from '../data/buildings.data';
import type { LogisticsAndTransportSystem } from './LogisticsAndTransportSystem';
import type { RoadNetworkSummary, RoadNetworkSystem } from './RoadNetworkSystem';
import { UtilityNetworkSystem } from './UtilityNetworkSystem';

export type DiagnosticCategory = 'all' | 'roads' | 'utilities' | 'vehicles';

export interface DiagnosticIssue {
  id: string;
  category: 'roads' | 'utilities' | 'vehicles';
  severity: 'critical' | 'warning';
  title: string;
  detail: string;
  resolutionHint: string;
  gx: number;
  gy: number;
  buildingId?: number;
}

export interface NetworkDiagnosticsReport {
  issues: DiagnosticIssue[];
  counts: {
    total: number;
    critical: number;
    warning: number;
    roads: number;
    utilities: number;
    vehicles: number;
  };
  roadSummary: RoadNetworkSummary;
  utilityHealth: {
    powerAvailableGen: number;
    powerDistCapacity: number;
    powerDemand: number;
    powerDelivered: number;
    waterAvailableGen: number;
    waterDistCapacity: number;
    waterDemand: number;
    waterDelivered: number;
  };
}

export class NetworkDiagnosticsSystem {
  public static analyze(
    state: WorldEngineState,
    chunks: ChunkManager,
    roadNet: RoadNetworkSystem,
    logistics: LogisticsAndTransportSystem
  ): NetworkDiagnosticsReport {
    roadNet.recomputeIfDirty(chunks, state);
    const issues: DiagnosticIssue[] = [];
    const seenIds = new Set<string>();

    const pushIssue = (issue: DiagnosticIssue) => {
      if (seenIds.has(issue.id)) return;
      seenIds.add(issue.id);
      issues.push(issue);
    };

    // 1. Disconnected road clusters & congested corridors (O(clusters + connectors))
    const clusterStats = roadNet.getAllClusterStats();
    const clusterRepTile = roadNet.getCachedClusterRepresentatives();
    const disconnectedConnectorTiles: Array<{ gx: number; gy: number; statusText: string }> = [];

    for (const ct of roadNet.getCachedConnectorTiles()) {
      const cs = roadNet.evaluateConnectorStatus(ct.gx, ct.gy, chunks);
      if (!cs.connected) {
        disconnectedConnectorTiles.push({ gx: ct.gx, gy: ct.gy, statusText: cs.statusText });
      }
    }

    for (const info of clusterStats.values()) {
      if (info.roadClass !== 'road' || !info.isDisconnected) continue;
      const rep = clusterRepTile.get(info.id) ?? { gx: 0, gy: 0 };
      pushIssue({
        id: `road_cluster_${info.id}`,
        category: 'roads',
        severity: info.buildings > 0 ? 'critical' : 'warning',
        title: `Disconnected Road Segment (#${info.id})`,
        detail: `${info.count} road ${info.count === 1 ? 'tile' : 'tiles'} isolated from the main road network${info.buildings > 0 ? ` (${info.buildings} buildings affected)` : ''}.`,
        resolutionHint: 'Link this segment to the main road network or connect border roads with a Connector.',
        gx: rep.gx,
        gy: rep.gy,
      });
    }

    for (const dc of disconnectedConnectorTiles.slice(0, 6)) {
      pushIssue({
        id: `connector_tile_${dc.gx}_${dc.gy}`,
        category: 'roads',
        severity: 'warning',
        title: `Connector Inactive at (${dc.gx}, ${dc.gy})`,
        detail: dc.statusText,
        resolutionHint: 'Attach an owned road on one side and locked-area forest road/frontier on the other.',
        gx: dc.gx,
        gy: dc.gy,
      });
    }

    // Top congested road corridors
    const hotRoads: Array<{ gx: number; gy: number; load: number }> = [];
    for (const [k, load] of roadNet.getTileCongestionMap().entries()) {
      if (load < 0.65) continue;
      hotRoads.push({ gx: unpackTileX(k), gy: unpackTileY(k), load });
    }
    hotRoads.sort((a, b) => b.load - a.load);
    for (const hr of hotRoads.slice(0, 4)) {
      pushIssue({
        id: `road_cong_${hr.gx}_${hr.gy}`,
        category: 'roads',
        severity: hr.load >= 0.85 ? 'critical' : 'warning',
        title: `Heavy Traffic Congestion (${Math.round(hr.load * 100)}%)`,
        detail: `Road bottleneck detected at (${hr.gx}, ${hr.gy}) slowing cargo & forestry trucks.`,
        resolutionHint: 'Upgrade to Two-Lane / Four-Lane / Highway or add bypass parallel routes.',
        gx: hr.gx,
        gy: hr.gy,
      });
    }

    // 2. Building-level Roads, Connector structures, Power & Water diagnostics
    for (const b of state.buildings) {
      if (b.status === 'constructing' || !b.enabled) continue;
      const def = BUILDINGS[b.defId];
      if (!def) continue;

      if (def.isConnectorStructure) {
        const cs = roadNet.evaluateConnectorBuildingStatus(b, chunks);
        if (!cs.connected) {
          pushIssue({
            id: `bldg_connector_${b.id}`,
            category: 'roads',
            severity: 'critical',
            title: `${def.name} Inactive`,
            detail: `${cs.statusText} at (${b.gx}, ${b.gy}).`,
            resolutionHint: 'Connect an owned road to any tile edge of this Connector.',
            gx: b.gx,
            gy: b.gy,
            buildingId: b.id,
          });
        }
      }

      const access = roadNet.getCachedRoadAccess(b, chunks);
      const needsHeavy =
        def.category === 'industrial' ||
        def.category === 'production' ||
        def.category === 'storage' ||
        def.subCategory === 'trade_port' ||
        (def.truckCapacity ?? 0) > 0;

      if (def.requiresRoad && access.accessPoints.length === 0) {
        pushIssue({
          id: `bldg_no_road_${b.id}`,
          category: 'roads',
          severity: 'critical',
          title: `${def.name}: No Road Access`,
          detail: b.roadReason || `Building at (${b.gx}, ${b.gy}) is not adjacent to any drivable road.`,
          resolutionHint: 'Build a road adjacent to the building footprint.',
          gx: b.gx,
          gy: b.gy,
          buildingId: b.id,
        });
      } else if (needsHeavy && access.restrictedForHeavyTrucks) {
        pushIssue({
          id: `bldg_heavy_road_${b.id}`,
          category: 'roads',
          severity: 'warning',
          title: `${def.name}: Heavy Truck Restricted`,
          detail: `Connected only to dirt/residential lanes at (${b.gx}, ${b.gy}); heavy freight trucks face speed & route penalties.`,
          resolutionHint: 'Upgrade adjacent road to Paved, Industrial, Two-Lane, or Four-Lane Road.',
          gx: b.gx,
          gy: b.gy,
          buildingId: b.id,
        });
      }

      if ((def.powerDemand ?? 0) > 0 && b.powerReason && b.powerReason !== 'ok') {
        const isCritical = !b.powerConnected || (b.powerSatisfaction ?? 0) <= 0.1;
        const hintMap: Record<string, string> = {
          not_connected_to_network: 'Run Power Lines from an active generator or place a Substation within 5 tiles.',
          disconnected_section_no_source: 'Wire this isolated power segment back to a running Power Plant / Turbine.',
          source_inactive: 'Fuel, staff, or enable the connected power generator.',
          distribution_overloaded: 'Add Substations or extra Power Lines to raise grid distribution capacity.',
          insufficient_supply: 'Build additional Power Generators or upgrade existing ones.',
        };
        pushIssue({
          id: `bldg_power_${b.id}`,
          category: 'utilities',
          severity: isCritical ? 'critical' : 'warning',
          title: `${def.name}: Power ${b.powerReason.replace(/_/g, ' ')}`,
          detail: `Grid delivery at ${Math.round((b.powerSatisfaction ?? 0) * 100)}% (${b.gx}, ${b.gy}).`,
          resolutionHint: hintMap[b.powerReason] ?? 'Check Power Grid overlay.',
          gx: b.gx,
          gy: b.gy,
          buildingId: b.id,
        });
      }

      if ((def.waterDemand ?? 0) > 0 && b.waterReason && b.waterReason !== 'ok') {
        const isCritical = !b.waterConnected || (b.waterSatisfaction ?? 0) <= 0.1;
        const hintMap: Record<string, string> = {
          not_connected_to_network: 'Lay Water Pipes from a Water Source or place a Water Pump within 5 tiles.',
          disconnected_section_no_source: 'Connect this isolated pipe segment to a working Water Tower or Pumping Station.',
          source_inactive: 'Staff or enable the connected Water Source.',
          distribution_overloaded: 'Add Water Pumps or parallel Pipes to boost pipe capacity.',
          insufficient_supply: 'Construct another Water Tower or Pumping Station.',
        };
        pushIssue({
          id: `bldg_water_${b.id}`,
          category: 'utilities',
          severity: isCritical ? 'critical' : 'warning',
          title: `${def.name}: Water ${b.waterReason.replace(/_/g, ' ')}`,
          detail: `Water flow at ${Math.round((b.waterSatisfaction ?? 0) * 100)}% (${b.gx}, ${b.gy}).`,
          resolutionHint: hintMap[b.waterReason] ?? 'Check Water Network overlay.',
          gx: b.gx,
          gy: b.gy,
          buildingId: b.id,
        });
      }
    }

    // 3. Orphaned power lines & water pipes (tiles without any source connected)
    const snap = UtilityNetworkSystem.getLastSnapshot();
    if (snap) {
      let orphanPowerReported = 0;
      for (const key of snap.powerTiles) {
        if (snap.powerTileNet.has(key)) continue;
        const gx = unpackTileX(key);
        const gy = unpackTileY(key);
        pushIssue({
          id: `orphan_power_${Math.floor(gx / 8)}_${Math.floor(gy / 8)}`,
          category: 'utilities',
          severity: 'warning',
          title: `Disconnected Power Line Section`,
          detail: `Power line at (${gx}, ${gy}) is not linked to any power generator.`,
          resolutionHint: 'Connect this wire run to a Solar Array, Wind Turbine, or Coal Power Plant.',
          gx,
          gy,
        });
        if (++orphanPowerReported >= 4) break;
      }

      let orphanWaterReported = 0;
      for (const key of snap.waterTiles) {
        if (snap.waterTileNet.has(key)) continue;
        const gx = unpackTileX(key);
        const gy = unpackTileY(key);
        pushIssue({
          id: `orphan_water_${Math.floor(gx / 8)}_${Math.floor(gy / 8)}`,
          category: 'utilities',
          severity: 'warning',
          title: `Disconnected Water Pipe Section`,
          detail: `Water pipe at (${gx}, ${gy}) has no active water source connected.`,
          resolutionHint: 'Link this pipe network to a Water Well, Water Tower, or Pumping Station.',
          gx,
          gy,
        });
        if (++orphanWaterReported >= 4) break;
      }
    }

    // 4. Vehicle & Forestry Logistics route diagnostics
    for (const rd of logistics.getRouteDiagnostics()) {
      pushIssue({
        id: `log_${rd.id}`,
        category: 'vehicles',
        severity: rd.severity,
        title: rd.title,
        detail: rd.detail,
        resolutionHint:
          rd.category === 'forestry'
            ? 'Ensure a Connector bridges owned roads to locked forest roads with permitted mature trees.'
            : 'Verify valid drivable road connectivity and widen bottlenecked segments.',
        gx: rd.gx,
        gy: rd.gy,
      });
    }

    for (const lt of logistics.loggingTrucks) {
      if (lt.state === 'blocked') {
        pushIssue({
          id: `logtruck_blocked_${lt.id}`,
          category: 'vehicles',
          severity: 'critical',
          title: `Logging Truck #${lt.id} Blocked`,
          detail: lt.statusReason ?? `Stalled near (${Math.round(lt.gx)}, ${Math.round(lt.gy)}).`,
          resolutionHint: 'Connect a Connector to an accessible locked forest road with mature trees.',
          gx: Math.round(lt.gx),
          gy: Math.round(lt.gy),
          buildingId: lt.factoryId,
        });
      }
    }

    for (const rv of logistics.roadVehicles) {
      if (rv.blockedSec >= 4.0) {
        pushIssue({
          id: `veh_stalled_${rv.id}`,
          category: 'vehicles',
          severity: 'warning',
          title: `Cargo Vehicle #${rv.id} Delayed`,
          detail: rv.statusReason ?? `Delayed by traffic or dead-end near (${Math.round(rv.gx)}, ${Math.round(rv.gy)}).`,
          resolutionHint: 'Check one-way lane directions or widen intersecting roads.',
          gx: Math.round(rv.gx),
          gy: Math.round(rv.gy),
          buildingId: rv.sourceBuildingId,
        });
      }
    }

    issues.sort((a, b) => {
      if (a.severity !== b.severity) return a.severity === 'critical' ? -1 : 1;
      return a.category.localeCompare(b.category);
    });

    const counts = {
      total: issues.length,
      critical: issues.filter((i) => i.severity === 'critical').length,
      warning: issues.filter((i) => i.severity === 'warning').length,
      roads: issues.filter((i) => i.category === 'roads').length,
      utilities: issues.filter((i) => i.category === 'utilities').length,
      vehicles: issues.filter((i) => i.category === 'vehicles').length,
    };

    return {
      issues,
      counts,
      roadSummary: roadNet.getNetworkSummary(logistics.roadVehicles.length),
      utilityHealth: {
        powerAvailableGen: state.powerAvailableGen ?? state.powerGenerated ?? 0,
        powerDistCapacity: state.powerDistCapacity ?? 0,
        powerDemand: state.powerDemand ?? 0,
        powerDelivered: state.powerDelivered ?? 0,
        waterAvailableGen: state.waterAvailableGen ?? state.waterGenerated ?? 0,
        waterDistCapacity: state.waterDistCapacity ?? 0,
        waterDemand: state.waterDemand ?? 0,
        waterDelivered: state.waterDelivered ?? 0,
      },
    };
  }
}
