/**
 * Where the use-case props act, in texture coordinates of uavid-000900.
 *
 * Measured against the photograph (brightness runs per row), not derived: the earlier version
 * picked "confident" cells automatically and marched the truck through a tree
 * row, which is the difference between an anchor and a measurement. The route
 * follows the eastern carriageway; the pad sits on the open plaza; the
 * beacons spread across the three search sectors.
 */

export type UV = [number, number];

export const DRONE_ANCHORS: UV[] = [[0.335, 0.72]];

export const ROUTE_ANCHORS: UV[] = [
  [0.9, 0.18],
  [0.85, 0.35],
  [0.8, 0.5],
  [0.755, 0.65],
  [0.715, 0.82],
];

export const RING_ANCHORS: UV[] = [
  // Open ground only: a search ring drawn over a tree row or a roof reads as
  // an obstacle inside the probe rather than a sector being swept.
  [0.315, 0.86],
  [0.60, 0.80],
  [0.845, 0.72],
];
