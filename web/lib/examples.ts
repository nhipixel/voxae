/**
 * Worked examples, mirroring the first five in gradio_app.EXAMPLE_QUERIES.
 *
 * All five are held-out evaluation samples, so each comes back scored against
 * annotated ground truth. Three share one frame.
 */

export type Example = {
  query: string;
  image: string;
  label: string;
  family: "affordance" | "referring";
};

export const EXAMPLES: Example[] = [
  {
    query:
      "Identify all ground surfaces, paved or otherwise cleared, that a heavy vehicle could drive across.",
    image: "/examples/uavid-000900.png",
    label: "Drivable surface",
    family: "affordance",
  },
  {
    query:
      "Find an open stretch of road big enough to stage equipment or park several vehicles side by side.",
    image: "/examples/uavid-000900.png",
    label: "Staging area",
    family: "affordance",
  },
  {
    query:
      "Show road surface that stays clear of overhanging tree canopy, suitable for a vehicle to idle without branch clearance issues.",
    image: "/examples/uavid-000900.png",
    label: "Clear of canopy",
    family: "affordance",
  },
  {
    query:
      "Mark the ground surfaces firm enough for a heavy vehicle to drive across, ignoring grass and trees.",
    image: "/examples/uavid-000400.png",
    label: "Firm ground",
    family: "affordance",
  },
  {
    query: "Highlight the large building block occupying the upper-left portion of the frame.",
    image: "/examples/uavid-000400.png",
    label: "Building block",
    family: "referring",
  },
];
