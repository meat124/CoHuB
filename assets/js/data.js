/*
 * Benchmark content, transcribed from the paper.
 * Tables 2, 3, 5, 6, 7, 11, 12, 13 and Figures 4, 5. Every chart and table on the
 * page is rendered from this file, so a number fixed here is fixed everywhere.
 */
window.COHUB_DATA = {
  // Task order follows Table 2. `collab` marks the collaborative phases of
  // Table 6 / Figure 4 (null for the three-humanoid tasks, which Figure 4 excludes).
  tasks: [
    {
      id: "handover", name: "Handover", robots: 2, movement: "low", coupling: "low",
      summary: "One humanoid transfers a bottle to its partner, which then places it at the target location.",
      scene: "Kitchen", sceneId: "Room/scene_084",
      objects: "Wine bottle; target pad",
      randomization: "Bottle and target pad independently: x, y ∈ [−0.025, 0.025].",
      episode: 0,
      phases: [
        { name: "Grasp and lift", collab: false, cond: "A’s right hand holds the bottle at least 3 cm above its counter resting height." },
        { name: "Transfer", collab: true, cond: "A’s and B’s right hands hold it together, followed by B’s right hand holding with both of A’s hands off, without an intervening all-hands release." },
        { name: "Place", collab: false, cond: "Released, stable, upright placement within ±6 cm of the target on each axis." }
      ],
      instructions: {
        Joint: "Pass the bottle directly from Robot A’s right hand to Robot B’s right hand, then place it on the target.",
        "Robot A": "Pick up the bottle directly in front of you with your right hand and hand it to your partner’s right hand.",
        "Robot B": "Receive the bottle with your right hand, place it on the target directly in front of you, and release it."
      }
    },
    {
      id: "pouring", name: "Pouring", robots: 2, movement: "low", coupling: "low",
      summary: "One humanoid pours a marble from a small cup into a glass held by its partner.",
      scene: "Kitchen", sceneId: "Room/scene_085",
      objects: "Cup, glass, and marble",
      randomization: "Cup and receiving glass independently: x ∈ [−0.03, 0.03], y ∈ [−0.06, 0.06], yaw ±30°; marble position within the cup: x, y ∈ [−0.006, 0.006].",
      episode: 0,
      phases: [
        { name: "Lift both vessels", collab: false, cond: "Cup and glass are simultaneously at least 3 cm above the worktop." },
        { name: "Tip", collab: false, cond: "A holds the cup aloft and tilts it at least 60°." },
        { name: "Pour", collab: true, cond: "The marble remains inside the glass continuously for 0.5 s, with the glass at least 3 cm above the worktop at completion. Returning the cup to the table is not required for success." }
      ],
      instructions: {
        Joint: "Pour the marble from the small cup into the glass the other robot is holding in the air.",
        "Robot A": "Take the squat drinking cup in front of you round its body, hold it over the mouth of your partner’s glass and tip it until the marble drops into the glass, then set the cup back on the table.",
        "Robot B": "Take the tall glass in front of you round its body, lift it off the table, and hold it out upright and still under your partner’s cup until the marble is in it."
      }
    },
    {
      id: "framehang", name: "FrameHang", robots: 2, movement: "low", coupling: "high",
      summary: "Two humanoids jointly lift and align a large picture frame and hang it on a wall-mounted peg.",
      scene: "Art gallery", sceneId: "Room/scene_136",
      objects: "Framed painting; wall peg",
      randomization: "Target peg and frame share x ∈ [−0.05, 0.05] and wall depth y ∈ [−0.025, 0.025]; target peg z ∈ [−0.04, 0]; extra frame offsets x ∈ [−0.03, 0.03], y ∈ [−0.01, 0.01].",
      episode: 37,
      phases: [
        { name: "Grasp", collab: false, cond: "Both robots hold the frame." },
        { name: "Lift", collab: true, cond: "They hold it together at least 10 cm above its spawn height." },
        { name: "Hang", collab: true, cond: "The peg passes through the frame’s loop while both robots hold the frame, which then remains threaded, stable, and released." }
      ],
      instructions: {
        Joint: "Lift the framed painting off the floor together, hang the loop on its top rail on the wall peg, and let go once it is hanging square.",
        "Robot A": "Take the left edge of the frame’s moulding with one hand, lift level with your partner, guide the loop onto the peg, and release once it is seated.",
        "Robot B": "Take the right edge of the frame’s moulding with one hand, lift level with your partner, guide the loop onto the peg, and release once it is seated."
      }
    },
    {
      id: "copouring", name: "CoPouring", robots: 2, movement: "low", coupling: "high",
      summary: "Two humanoids lift a tub of apples together and tilt it to pour the apples into an empty crate.",
      scene: "Grocery store", sceneId: "Room/scene_102",
      objects: "Tub of apples; receiving crate",
      randomization: "Tub: x ∈ [−0.02, 0.02], y ∈ [−0.01, 0.01]; receiving crate (target), independently: x ∈ [−0.06, 0.06], y ∈ [−0.005, 0.005]; apples relative to tub: x, y ∈ [−0.008, 0.008]. Supporting stacks follow their containers.",
      episode: 53,
      phases: [
        { name: "Joint grasp and lift", collab: true, cond: "Both robots hold the tub at least 3 cm above its stack." },
        { name: "Tip", collab: true, cond: "They hold it together at a tilt of at least 25°." },
        { name: "Pour", collab: true, cond: "All apples are inside the receiving crate, with both robots still holding the tub." }
      ],
      instructions: {
        Joint: "Lift the tub of apples together and pour all of them into the empty crate beside you.",
        "Robot A": "Take the bar at your end of the tub with one hand, lift it level with your partner, and tip it together into the empty crate on your left until all the apples are in it.",
        "Robot B": "Take the bar at your end of the tub with one hand, lift it level with your partner, and tip it together into the empty crate on your right until all the apples are in it."
      }
    },
    {
      id: "trashcollection", name: "TrashCollection", robots: 2, movement: "high", coupling: "low",
      summary: "One humanoid carries a bin to the end of a table, where its partner sweeps a bottle into it.",
      scene: "Science laboratory", sceneId: "Room/scene_147",
      objects: "Bin; plastic bottle on bench",
      randomization: "+y toward Robot A. Bin: x ∈ [−0.15, 0.15], y ∈ [−0.15, −0.04]. Bottle: x ∈ [−0.10, 0.10], y ∈ [−0.04, 0.12].",
      episode: 56,
      phases: [
        { name: "Lift bin", collab: false, cond: "A holds the bin at least 2 cm off the floor." },
        { name: "Deliver bin", collab: false, cond: "A holds it aloft within 25 cm of the bench-side carry point, with its rim no more than 1 cm above the bench top." },
        { name: "Sweep", collab: true, cond: "The bottle is inside the bin, with A still holding the bin." }
      ],
      instructions: {
        Joint: "Bring the bin to the bench and sweep the bottle off the bench into it.",
        "Robot A": "Bend and take the tall bin in both hands, carry it upright straight ahead to the end of your partner’s bench, and hold it there with its rim just below the bench top.",
        "Robot B": "Wait at the bench. When the bin is at the end of it, sweep the plastic bottle lying between your hands along the surface and over that end into the bin; it rolls, so push straight and keep it off the near edge."
      }
    },
    {
      id: "cartservice", name: "CartService", robots: 2, movement: "high", coupling: "low",
      summary: "One humanoid pushes a trolley to a table, while its partner retrieves a wine bottle from the trolley and places it on the table.",
      scene: "Restaurant", sceneId: "Room/scene_110",
      objects: "Service trolley; wine bottle",
      randomization: "+y toward Robot A. Trolley and its contents: x ∈ [−0.10, 0.10], y ∈ [−0.30, 0.08]. Goal: y ∈ [−0.16, −0.12].",
      episode: 26,
      // Appendix B.2: the only two-humanoid task without a collaborative phase.
      phaseNote: "All phases are non-collaborative, but the task as a whole is collaborative because Robot B retrieves the bottle from the trolley that Robot A delivers.",
      phases: [
        { name: "Park", collab: false, cond: "The trolley center reaches within 35 cm of the parking mark." },
        { name: "Lift bottle", collab: false, cond: "B lifts the bottle from the trolley positioned by A to at least 10 cm above its spawn height." },
        { name: "Serve", collab: false, cond: "B carries it within 15 cm of the goal in the horizontal plane, then releases it in a stable, upright placement within ±6 cm of the goal on each axis." }
      ],
      instructions: {
        Joint: "Push the loaded service trolley down the aisle to the green mark on the floor, then take the wine off it and stand the bottle on the green patch on the table.",
        "Robot A": "Take the push bar of the trolley in front of you with both hands and push it straight down the aisle until it stands on the green rectangle painted on the floor.",
        "Robot B": "Walk up the aisle to meet the trolley and stand square in front of its near edge; take the wine bottle around the glass just above the steel ring it stands in, lift it straight up out of the ring, then step round to your right to the near edge of the table and stand it on the green patch."
      }
    },
    {
      id: "cocarry", name: "CoCarry", robots: 2, movement: "high", coupling: "high",
      summary: "Two humanoids jointly carry a laundry basket to a target location while keeping it balanced.",
      scene: "Laundromat", sceneId: "Room/scene_143",
      objects: "Laundry basket; goal shelf",
      randomization: "Basket: x, y ∈ [−0.05, 0.05].",
      episode: 5,
      phases: [
        { name: "Joint grasp and lift", collab: true, cond: "Both robots hold the basket with its base at least 3 cm above the shelf." },
        { name: "Carry", collab: true, cond: "Both hold it within the goal’s ±15 cm horizontal bounds, with at least 1 cm base clearance above the shelf." },
        { name: "Release", collab: true, cond: "Stable placement within those bounds and ±6 cm of the target height, with all hands off." }
      ],
      instructions: {
        Joint: "Carry the laundry basket together and set it down level on the green patch on the far shelf.",
        "Robot A": "Hold your side of the basket with both hands and side-step to your right, keeping it level, until it rests on the green patch.",
        "Robot B": "Hold your side of the basket with both hands and side-step to your left, keeping it level, until it rests on the green patch."
      }
    },
    {
      id: "tablealign", name: "TableAlign", robots: 2, movement: "high", coupling: "high",
      summary: "Two humanoids carry a table across an office, rotate it during transport, and place it aligned with a chair.",
      scene: "Open-plan office", sceneId: "Room/scene_115",
      objects: "Table; fixed goal chair",
      randomization: "Table: x, y ∈ [−0.05, 0.05].",
      episode: 56,
      phases: [
        { name: "Joint grasp and lift", collab: true, cond: "Both robots hold the table at least 3 cm above its standing height." },
        { name: "Carry", collab: true, cond: "Both hold it inside the goal region with every foot bar at least 1 cm off the floor." },
        { name: "Align", collab: true, cond: "Released, stable placement with the table center 33.5–46.5 cm in front of the chair, within ±13 cm laterally and ±4 cm of standing height, and square to the chair within 4°." }
      ],
      instructions: {
        Joint: "Carry the table to the chair together and set it down square to it.",
        "Robot A": "Take your end of the table with both hands, lift level with your partner, and side-step to your right down the room, turning the table a quarter circle together as you go, until it stands on the green patch with the chair tucked into its long edge.",
        "Robot B": "Take your end of the table with both hands, lift level with your partner, and side-step to your left down the room, turning the table a quarter circle together as you go, until it stands on the green patch with the chair tucked into its long edge."
      }
    },
    {
      id: "movehouse", name: "MoveHouse", robots: 3, movement: null, coupling: null,
      summary: "One humanoid opens a door while the other two carry a table through the doorway and place it at the target location.",
      scene: "Home: bedroom and living room", sceneId: "House/scene_098",
      objects: "Console table; lever-handle door",
      randomization: "Table: x, y ∈ [−0.05, 0.05]; door initially closed and latched.",
      episode: 29,
      phases: [
        { name: "Lift and open", collab: null, cond: "B and C jointly lift every table foot at least 3 cm off the floor; A opens the door to at least 1.45 rad (≈ 83°) for 0.5 s. These two milestones may occur in either order." },
        { name: "Pass doorway", collab: null, cond: "Table and both carriers pass the doorway clearance plane together, with B and C holding and the feet off the floor." },
        { name: "Place", collab: null, cond: "B and C carry the table within 15 cm of the goal while holding it aloft, then release it stably with each foot within its 10 cm-square mark and height within ±4 cm of standing height." }
      ],
      instructions: {
        Joint: "One robot opens the door and stands clear while the other two carry the table out of the bedroom, through the doorway, and into the living room.",
        "Robot A": "Walk to the door, take the lever, push the door wide open, and stand out of the lane so your partners and their table can pass.",
        "Robot B": "Take the two legs at your end of the table, one in each hand, lift level with your partner, and lead it backwards through the open doorway until its four legs stand on the green squares.",
        "Robot C": "Take the two legs at your end of the table, one in each hand, lift level with your partner, and follow them forwards through the open doorway until its four legs stand on the green squares."
      }
    },
    {
      id: "bigtable", name: "BigTable", robots: 3, movement: null, coupling: null,
      summary: "All three humanoids jointly lift a large round table, carry it to the target location, and set it down together.",
      scene: "Dining room", sceneId: "Room/scene_046",
      objects: "Round pedestal table",
      randomization: "Table: x, y ∈ [−0.04, 0.04], yaw ±4°.",
      episode: 14,
      phases: [
        { name: "Three-robot grasp and lift", collab: null, cond: "All three robots hold the table with the pedestal foot at least 3 cm off the floor." },
        { name: "Carry", collab: null, cond: "All three hold it with the entire pedestal foot over the goal disc and at least 1 cm floor clearance." },
        { name: "Release", collab: null, cond: "The pedestal foot remains within the disc, with stable, upright, released placement within ±5 cm of standing height." }
      ],
      instructions: {
        Joint: "All three robots lift the round table together and carry it to the marked spot.",
        "Robot A": "Take the edge of the table in front of you with both hands, lift on the count, and walk it straight forward to the green disc, keeping it level.",
        "Robot B": "Take the edge of the table in front of you with both hands, lift on the count, and walk it backwards and to your right to the green disc, keeping it level.",
        "Robot C": "Take the edge of the table in front of you with both hands, lift on the count, and walk it backwards and to your left to the green disc, keeping it level."
      }
    }
  ],

  // Hero mosaic, rows of 4 / 2 / 4 as in Figure 1 (three-humanoid tasks in the middle).
  heroLayout: {
    top: ["framehang", "copouring", "handover", "pouring"],
    middle: ["movehouse", "bigtable"],
    bottom: ["trashcollection", "cartservice", "tablealign", "cocarry"]
  },

  // Opening shot: this task plays full screen for `holdMs`, then shrinks into its
  // mosaic cell while the other nine tasks appear. `src` is a higher-resolution
  // copy used only for this tile.
  heroIntro: { task: "movehouse", holdMs: 4200, src: "assets/video/intro/movehouse.mp4" },

  families: [
    { id: "sil", name: "Standard IL", long: "Standard IL policies", training: "per task" },
    { id: "mail", name: "Multi-agent IL", long: "Multi-agent IL policies", training: "per task" },
    { id: "vla", name: "VLA", long: "Vision-Language-Action (VLA) models", training: "multi-task" },
    { id: "wam", name: "WAM", long: "World Action Models (WAMs)", training: "multi-task" }
  ],

  // `sub` renders as a subscript (π0.5, Ψ0).
  methods: [
    { id: "act", name: "ACT", family: "sil" },
    { id: "dp", name: "DP", family: "sil" },
    { id: "latenttom", name: "LatentToM", family: "mail", note: "Evaluated only on tasks with two humanoids, following the original paper’s setup." },
    { id: "gaudp", name: "GauDP", family: "mail" },
    { id: "pi05", name: "π", sub: "0.5", family: "vla" },
    { id: "gr00t", name: "GR00T N1.7", family: "vla" },
    { id: "psi0", name: "Ψ", sub: "0", family: "vla" },
    { id: "fastwam", name: "Fast-WAM", family: "wam" }
  ],

  // Tables 11–13: cumulative phase success (%) [p1, p2, p3]; p3 is full-task
  // success and equals Table 2. null = not evaluated.
  phaseResults: {
    act:       { handover: [100, 98, 65], pouring: [95, 93, 1], framehang: [100, 100, 44], copouring: [33, 32, 9],  trashcollection: [52, 16, 1], cartservice: [69, 33, 3], cocarry: [99, 72, 61], tablealign: [100, 79, 39], movehouse: [89, 80, 9],  bigtable: [100, 98, 80] },
    dp:        { handover: [87, 64, 30],  pouring: [38, 38, 0], framehang: [76, 73, 26],   copouring: [21, 18, 7],  trashcollection: [32, 4, 1],  cartservice: [66, 9, 3],  cocarry: [96, 31, 12], tablealign: [93, 34, 4],   movehouse: [73, 57, 4],  bigtable: [98, 50, 14] },
    latenttom: { handover: [77, 23, 5],   pouring: [48, 41, 0], framehang: [58, 24, 2],    copouring: [13, 10, 1],  trashcollection: [30, 5, 0],  cartservice: [52, 6, 0],  cocarry: [98, 44, 31], tablealign: [96, 23, 3],   movehouse: null,         bigtable: null },
    gaudp:     { handover: [92, 85, 52],  pouring: [29, 28, 0], framehang: [81, 76, 29],   copouring: [0, 0, 0],    trashcollection: [52, 6, 1],  cartservice: [64, 2, 0],  cocarry: [92, 22, 9],  tablealign: [95, 62, 16],  movehouse: [38, 30, 2],  bigtable: [100, 83, 54] },
    pi05:      { handover: [92, 12, 2],   pouring: [58, 54, 0], framehang: [91, 86, 27],   copouring: [1, 1, 0],    trashcollection: [56, 10, 0], cartservice: [60, 34, 4], cocarry: [73, 4, 1],   tablealign: [75, 6, 2],    movehouse: [33, 17, 3],  bigtable: [77, 44, 37] },
    gr00t:     { handover: [95, 77, 27],  pouring: [63, 59, 0], framehang: [94, 90, 44],   copouring: [2, 2, 0],    trashcollection: [49, 13, 1], cartservice: [77, 36, 0], cocarry: [49, 7, 2],   tablealign: [62, 6, 1],    movehouse: [40, 22, 0],  bigtable: [36, 9, 5] },
    psi0:      { handover: [65, 8, 0],    pouring: [24, 18, 0], framehang: [62, 53, 5],    copouring: [1, 0, 0],    trashcollection: [55, 18, 3], cartservice: [77, 2, 0],  cocarry: [82, 0, 0],   tablealign: [77, 0, 0],    movehouse: [61, 45, 0],  bigtable: [87, 23, 19] },
    fastwam:   { handover: [98, 62, 2],   pouring: [86, 84, 7], framehang: [97, 94, 28],   copouring: [28, 25, 11], trashcollection: [55, 11, 1], cartservice: [73, 45, 0], cocarry: [99, 53, 31], tablealign: [96, 40, 17],  movehouse: [95, 80, 20], bigtable: [97, 94, 69] }
  },

  // Figure 4: average success (%) on non-collaborative vs collaborative phases,
  // 8 two-humanoid tasks, 50 demonstration-derived initializations per phase.
  collabGap: [
    { method: "act", nc: 86.2, c: 72.1, drop: -16.3 },
    { method: "dp", nc: 72.0, c: 65.0, drop: -9.7 },
    { method: "latenttom", nc: 63.4, c: 56.1, drop: -11.4 },
    { method: "gaudp", nc: 69.8, c: 60.6, drop: -13.2 },
    { method: "pi05", nc: 76.6, c: 52.4, drop: -31.6 },
    { method: "gr00t", nc: 73.4, c: 53.0, drop: -27.8 },
    { method: "psi0", nc: 54.8, c: 46.4, drop: -15.3 },
    { method: "fastwam", nc: 76.8, c: 62.7, drop: -18.3 }
  ],

  // GPT-6 Astra (appendix, Astra phase-wise tables): evaluated through Codex without
  // benchmark-specific training, 10 rollouts per task, two-humanoid tasks only.
  // Phase success (%) [p1, p2, p3]; p3 is full-task success.
  astra: {
    name: "GPT-6 Astra",
    rollouts: 10,
    phases: {
      handover: [10, 0, 0], pouring: [90, 80, 10], framehang: [100, 60, 0], copouring: [100, 70, 70],
      trashcollection: [90, 0, 0], cartservice: [70, 0, 0], cocarry: [40, 0, 0], tablealign: [80, 10, 0]
    }
  },

  // Table 3: average task success (%) over the 8 two-humanoid tasks.
  designChoices: {
    models: ["act", "dp", "gr00t"],
    rows: [
      { scope: "Local", policy: "Separate", values: [27.9, 10.4, 14.5] },
      { scope: "Local", policy: "Shared", values: [23.9, 9.5, 9.4] },
      { scope: "Global", policy: "Separate", values: [25.6, 19.1, 12.3] },
      { scope: "Global", policy: "Shared", values: [26.4, 16.0, 9.0] }
    ]
  },

  // Figure 5 tasks that have a full-rollout video (assets/video/failures/<id>.mp4, re-rendered
  // from the logged simulator state); the rest still show the two paper stills.
  failureVideos: ["handover", "pouring", "framehang", "copouring", "trashcollection",
    "cartservice", "cocarry", "tablealign", "movehouse", "bigtable"],

  // Figure 5 (GR00T N1.7 rollouts), captions verbatim from the paper.
  failures: {
    handover: "The giver lets go before the receiver’s hand closes on the bottle, which drops between the two hands.",
    pouring: "The cup is tipped while the partner’s glass is not beneath it, so the marble misses the glass.",
    framehang: "The robots lift the frame at mismatched heights and angles, so its loop misses the peg and the frame slips.",
    copouring: "The robots fail to synchronize their grasps, leaving one handle ungrasped and causing the apples to spill outside the receiving crate.",
    trashcollection: "The bottle is swept off the laboratory table but misses the bin and falls to the floor.",
    cartservice: "The receiving robot carries the bottle to the goal table but walks past it instead of placing it.",
    cocarry: "One robot loses its grip on the basket, unbalancing the load and causing it to fall.",
    tablealign: "One robot loses its grip during transport, and the table falls outside the target region.",
    movehouse: "The two robot carriers fail to coordinate their motion, causing the table to deviate from the doorway path and get stuck against the door frame.",
    bigtable: "During transport, one robot loses its grip on the table, which tilts and never reaches the goal."
  }
};
