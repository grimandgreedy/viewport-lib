# Showcase Recommendations

This is a companion to `CATALOGUE.md`. It groups the 51 showcases, flags overlap, proposes consolidations and removals, suggests ordering, and reflects on the "what's the theme?" question. Read alongside the catalogue rather than in isolation.

The premise: viewport-lib has outgrown its original showcase set. Several demos were carved out when a feature was new and minor; the same feature is now one toggle inside a larger demo somewhere else. Headers don't match UIs. The same item type is exercised from three places. There's no through-line a new user can follow.

Concrete proposals follow. None of them are "do this all at once" — they're a menu.

---

## 1. The thematic question first

You said: *"they just demo cool stuff, there is no uniform theme; again, maybe that is a good thing but worth thinking about."*

There are two stances and they aren't both right or both wrong; they trade off.

**Stance A — keep it a cool-stuff tour.** The current showcase set works like Three.js's examples page: discrete, visually distinct demos, each one is a "what if I want to do X" answer. New users browse for an idea that looks like theirs. This is genuinely useful and is the rarer thing — most engines bury this stuff under tutorials. If you go this way, the work is **menu clustering + better headers + cross-references**, not consolidation.

**Stance B — restructure around capability domains.** Fewer, denser showcases, each exhaustively covering one area (Lighting, Scientific Viz, Animation, etc.). Loses the "wow factor" of discrete demos and creates one-paragraph-each combined panels that nobody reads top-to-bottom. This is the typical mature-library outcome and it's worse than what you have today.

**Recommendation: Stance A with discipline.** Keep the discrete demos, but (i) cluster them in the menu so the discovery path is obvious, (ii) drop / fold the handful that are genuinely redundant, (iii) fix the header drift (it's currently masking a real onboarding problem). Don't restructure for the sake of restructuring.

The rest of this document is concrete suggestions in service of that stance.

---

## 2. Removal / fold candidates (highest-confidence wins)

These are the ones where the cost of having the showcase exceeds what it teaches that isn't already taught better elsewhere.

### Strong candidates to fold

**9. Annotations → fold into 34. Labels.** Both are `LabelItem` demos. 34 is the comprehensive one (every label knob + a realistic gearbox use-case). 9 is a 129-line read-only diagnostic page that only adds the visible / clipped / screen-anchored classification — which could be a third checkbox on 34. Net: -1 menu entry, no capability loss.

**10. Camera Tools → fold into 27. Camera Framing & HUD.** 10 is the smallest "still has interactivity" showcase (146 lines, just `ViewPreset` buttons + projection + FOV). 27 already has fly-to and turntable. Adding the view-preset grid as a fourth section of 27 is a 30-line change. Net: -1 menu entry.

**32. Extended Quantities → split across its three natural homes.** 32 is explicitly a leftover-collector for quantity types that didn't fit elsewhere when they were added:
- 32A (Edge / Halfedge / Corner scalars) → belongs in **20. Face Attributes** as additional kinds, or **12. Scalar Fields** as additional objects.
- 32B (Volume mesh vector arrows) → belongs in **26. Volume Meshes** as a "show radial vectors" toggle.
- 32C (Point cloud per-point radius + transparency) → belongs in **15. Point Clouds & Glyphs** as a fourth sub-mode (and 15's `PointGaussian` already overlaps).

Once split, delete 32. Net: -1 menu entry, three richer showcases.

**40. GPU Vertex Warp → fold into 50. GPU Wave.** 40 (201 lines, one slider) demonstrates `warp_attribute` / `warp_scale` with three baked displacement fields. 50 demonstrates a chained GPU plugin overriding mesh positions. They cover the same conceptual ground (GPU-driven vertex displacement) at different layers. 50 can carry an extra sub-mode showing the baked-attribute path for comparison. Net: -1 menu entry.

**21. Textures → fold into 7. Normal Maps or 19. Matcap.** 21 has zero interactive controls — it's a static page of "here are four textured objects". The Percy photo is charming but doesn't teach anything 7 doesn't already cover. Either fold the Percy plane into 7 as an extra object, or accept 21 as "the marketing screenshot" and leave it alone with a one-line label change ("static textures gallery"). Net: -1 menu entry if folded.

### Weaker candidates (worth thinking about but less obvious)

**29. Depth-Composited Images.** Niche but cleanly demonstrates `ScreenImageItem::depth`. It's small and self-contained. Keep but accept it's a footnote.

**44. Debug Draw.** 327 lines, demonstrates `DebugDraw` and physics contact-event reading. Could fold into 43 (Scene Runtime) as a third demo since both use `PhysicsLitePlugin`. Smaller win than the others above.

---

## 3. Combine / restructure candidates

### Lighting cluster (4 showcases, lots of overlap)

- **8. Shadows** — CSM cascade count + PCF/PCSS + contact shadows
- **11. Lights** — add/remove dynamic lights + hemisphere + EDL
- **47. Lighting Consistency** — broadcast `ItemSettings` flags across every item type
- **49. Scene Lights** — `Scene::add_light` API + cluster fallback under stress

These four are distinct and each justifies its own slot, but the **menu** should group them together (today they're spread across slots 8, 11, 47, 49). 49's Stress tab also overlaps 23 (Performance) in that both are "throw a lot of stuff at the renderer and watch stats."

Proposal: keep all four. Cluster them in the menu as "Lighting." Optionally fold 47 into the menu next to 11 (they're the two "edit lights, see effect" demos; 47 just adds the "broadcast `ItemSettings`" axis).

### Volumes / volume meshes cluster

- **17. Volume & Isosurface** — ray-march volume + marching cubes + image+surface slices
- **18. Clip Volumes** — `ClipObject` against torus+capsule + density volume version
- **26. Volume Meshes** — unstructured volume meshes (Hex/Tet/Pyramid/Wedge)
- **30. Implicit Surfaces** — five rendering paths for SDF blobs (CPU/GPU sphere-march + MC + GPU implicit + GPU MC)
- **31. Sparse Volume Grid** — sparse-grid topology + interactive paint cube
- **39. Tensor Glyphs** — sits on a volume mesh but is really about tensors

This is a busy area. They're all worth keeping (each demonstrates a distinct item type or extraction path), but they need menu clustering. The two with the most overlap are **17** and **30** (both do isosurface extraction; both have a "GPU vs CPU" toggle). They're worth keeping separate because 17 is "volumetric data" and 30 is "implicit functions", but the line is blurry and a casual reader won't see the distinction.

### Curves / lines / streams cluster

- **14. Isolines & Contours** — isoline strips on a scalar surface
- **16. Streamlines & Tubes** — polyline/streamtube/tube/ribbon as four render modes
- **28. Curve Network Quantities** — six per-node/per-edge `PolylineItem` quantity attributes

Each is distinct, but 28 is essentially "the rest of the PolylineItem API surface that 16 doesn't cover." Plausible merge: fold 28's six modes as additional render modes in 16 (it already has four). Net: -1 menu entry, but you'd be cramming nine sub-modes into one showcase, which works against discoverability.

Probably better: keep separate, cluster in menu under "Curves & Lines."

### Animation / runtime cluster

- **36. Playback Runtime Control** — RuntimeMode + PerformancePolicy stress test
- **43. Scene Runtime** — `ViewportRuntime` + `RuntimePlugin` + physics + animation
- **44. Debug Draw** — debug primitives written from a runtime plugin
- **45. Skeletal Animation** — `Skeleton` / `Pose` / glTF / crowd
- **50. GPU Wave** — `GpuPlugin` compute path

These all use the runtime layer. They're separable because each demonstrates a different runtime entry point. Menu clustering rather than consolidation.

---

## 4. Header drift — half the showcases are misleading

Per the catalogue, the following have significant drift between their `//!` doc-comment and their actual UI:

`2, 3, 4, 6, 7, 11, 15, 16, 17, 26, 30, 31, 33, 35, 39, 41, 45, 46, 48, 50, 51`

That's 21 of 51. Many of these (11, 15, 16, 17, 26, 30, 31, 33, 35, 41, 45, 51) grew sub-modes after the header was written. A few (6, 35, 46, 51) actively contradict the header.

This is the biggest tractable improvement. The fix is not deep:

1. For each drifting showcase, rewrite the `//!` block to describe what the UI does today.
2. If the showcase has sub-modes, list every sub-mode by name in the header.
3. If the header makes a "tracked elsewhere" or "land in J5" promise that's been fulfilled or abandoned, remove the promise.

This costs maybe a day of focused work and dramatically improves what someone sees when they grep the showcase files looking for an example.

The catalogue you now have is in effect the cleaned-up header set; some of it could be lifted directly into each showcase's `//!`.

---

## 5. Ordering — current vs proposed

Current menu order (numbers from `main.rs`):

> 1 Basic → 2 Scene → 23 Performance → 4 Interaction → 5 Materials → 6 Post-FX → 7 Normal → 8 Shadows → 9 Annotations → 10 Camera → 11 Lights → 12 Scalar → 13 Multi-view → 14 Isolines → 15 Points → 16 Streams → 17 Volume → 18 Clip → 19 Matcap → 20 Face Attr → 21 Tex → 22 UV → 3 Ground → 24 Backface → 25 Surface Vec → 26 Volume Mesh → 27 Camera Frame → 28 Curve Net → 29 Depth-comp → 30 Implicit → 31 Sparse → 32 Ext Quant → 33 Pick → 34 Labels → 35 Overlay → 36 Playback → 37 Probe → 38 LIC → 39 Tensor → 40 Warp → 41 Sprites → 42 Splats → 43 Runtime → 44 Debug Draw → 45 Skinned → 46 Decals → 47 Consistency → 48 Scatter → 49 Scene Lights → 50 GPU Wave → 51 Async

That's the "in the order they were added, with three small swaps." 23 jumps to slot 3, 3 jumps to slot 23. A new user has no chance of finding what they need by scanning this.

### Proposed menu clustering

This is what I'd put in the sidebar dropdown if the goal is "a new user can find the right showcase in 10 seconds." Showcase numbers stay the same (no file renames needed) — only the menu groups change.

**Getting started**
1. Rendering Basics
2. Scene Graph
5. Materials and Visibility
22. UV Parameterization
24. Backface Policy

**Cameras & interaction**
4. Interaction (gizmo / G/R/S / view presets / zoom-to-fit / spline widget)
10. Camera Tools  *(fold into 27)*
13. Multi-Viewport
27. Camera Framing & HUD (fly-to / turntable / track)
37. Probe Widgets

**Lighting & shading**
3. Ground Plane
6. Post-Processing
7. Normal Maps & AO
8. Shadows
11. Lights
19. Matcap
47. Lighting Consistency
49. Scene Lights

**Textures & decals**
21. Textures  *(fold into 7 or treat as gallery)*
46. Decals

**Scientific visualization — scalar & vector quantities**
12. Scalar Fields
14. Isolines & Contours
15. Point Clouds & Glyphs
16. Streamlines & Tubes
20. Face Attributes
25. Surface Vectors
28. Curve Network Quantities
32. Extended Quantities  *(split across 12 / 26 / 15)*
38. Surface LIC
39. Tensor Glyphs

**Volumes & implicit surfaces**
17. Volume & Isosurface
18. Clip Volumes
26. Volume Meshes
30. Implicit Surfaces
31. Sparse Volume Grid

**Annotation & overlay**
9. Annotations  *(fold into 34)*
29. Depth-Composited Images
34. Labels
35. Overlay Composition

**Particles & sprites**
41. Sprites & Particles
42. Gaussian Splats

**Animation, runtime & GPU plugins**
36. Playback Runtime Control
40. GPU Vertex Warp  *(fold into 50)*
43. Scene Runtime
44. Debug Draw
45. Skeletal Animation
50. GPU Wave

**Picking & performance**
23. Performance
33. Picking Levels
51. Async Asset Streaming

### What this changes for the user

- Someone looking for "how do I show a vector field" sees four candidates in one cluster (15, 16, 25, 38) instead of hunting across the whole menu.
- Lighting becomes a self-contained tutorial path (3 → 6 → 7 → 8 → 11 → 19 → 47 → 49) instead of scattered demos.
- The scientific-viz coverage is visibly substantial — it's currently easy to miss how much of the library this is.
- The "I want to learn the runtime layer" path becomes a single cluster (36, 43, 44, 45, 50).

### What this doesn't change

- Numbers and file names stay the same. No rename churn.
- Anyone who already knows "I want showcase 17" still finds it instantly.
- The "this is a cool-stuff tour" property is preserved; only the index changes.

---

## 6. Suggested next steps in priority order

1. **Fix the headers.** 21 showcases drift. This is the highest ratio of user-visible improvement to effort.
2. **Implement the menu clustering.** Add section dividers / headings to the showcase selector dropdown. No code moves.
3. **Fold the four clearest cases.** 9 → 34, 10 → 27, 32 → split, 40 → 50. About a day of work each.
4. **Decide on 21.** Either fold into 7 or accept as a static gallery with a one-line label change.
5. **Optionally**: 47 + 11 menu adjacency; 44 + 43 menu adjacency; ride out the rest.

Don't restructure further. The discreteness is a feature.
