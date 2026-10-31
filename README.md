<h1 align="center">diffgrasp</h1>

<p align="center">
  <b>Differentiable grasp synthesis for a procedural 21-DOF human hand, in PyTorch.</b>
</p>

<p align="center">
  <a href="https://github.com/Dy07077/diffgrasp/actions/workflows/ci.yml"><img alt="tests" src="https://github.com/Dy07077/diffgrasp/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

<p align="center"><img src="assets/grasp.gif" width="62%" alt="a hand closing around a mug during optimisation"></p>

Describe an object as a signed distance function, and diffgrasp optimises dozens of hand poses in
parallel by gradient descent. The grasps it reports touch the surface, sink less than 1 mm into it,
and can resist any small disturbance under Coulomb friction. That last property is force closure.
Both are verified independently of the optimiser: force closure with the Ferrari–Canny metric, and
penetration with a dense sampled check in both directions.

## Gallery

<p align="center"><img src="assets/gallery.png" width="100%" alt="best grasp for six objects"></p>

| Object | Grasp | Valid seeds (95% CI) | Median ε, valid seeds | Shown grasp: ε | Contacts | Penetration | Time (48 seeds) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ball | precision | 98% (89–100%) | 0.087 | 0.081 | 3 | 0.06 mm | 24 s |
| cube | precision | 33% (22–47%) | 0.060 | 0.094 | 3 | 0.22 mm | 25 s |
| donut | power | 65% (50–77%) | 0.069 | 0.140 | 10 | 0.32 mm | 47 s |
| can | power | 58% (44–71%) | 0.083 | 0.084 | 11 | 0.31 mm | 49 s |
| mug | power | 46% (33–60%) | 0.064 | 0.089 | 11 | 0.52 mm | 59 s |
| bottle | power | 60% (46–73%) | 0.038 | 0.063 | 11 | 0.25 mm | 59 s |

A grasp counts as **valid** when it is force-closure with μ = 0.5 (Ferrari–Canny ε > 0), has at least
three contacts within 2 mm of the surface, and penetrates less than 1 mm. ε measures the largest
disturbance wrench the grasp can resist per unit contact force, with torques divided by the object
radius. The picture shows the lowest-energy valid seed of each object.

Penetration comes from `dg.penetration_depth`, which is independent of the samples the optimiser
uses and checks both directions: about 18k points on the hand against the object's exact SDF, and
30k points on the object against the hand's capsules and palm hull. The second direction catches an
edge or a thin part poking into the hand between hand samples.

How many of the 48 seeds fail each check:

| Object | Fewer than 3 contacts within 2 mm | Not force closure | Penetration ≥ 1 mm |
| --- | ---: | ---: | ---: |
| ball | 1 | 1 | 0 |
| cube | 31 | 32 | 0 |
| donut | 3 | 15 | 7 |
| can | 8 | 20 | 0 |
| mug | 19 | 26 | 1 |
| bottle | 10 | 19 | 0 |

Each check is counted on its own, so one seed can appear in several columns. Force closure is
evaluated only on touching contacts, so every seed with fewer than three of them fails it too. Most
failures stop a few millimetres short of the surface, or close around the object without reaching
force closure. Penetration rejects 8 of the 288 seeds, all of them deep: in 7 donut seeds the palm
ends up inside the tube, 16.4–16.8 mm deep, and in one mug seed part of the mug passes 14.4 mm into
the palm. Two of the eight fail on penetration alone. `assets/results.json` keeps every seed's
result.

<p align="center"><img src="assets/contact_maps.png" width="100%" alt="contact maps on the hand"></p>

## How it works

**Hand.** A kinematic tree with 21 joint angles: four per finger (MCP abduction and flexion, PIP,
DIP) and five for the thumb. The thumb's CMC joint includes the axial rotation that lets its pad
face the fingers in opposition. A sigmoid maps unconstrained parameters into anatomical joint
limits, and the wrist pose uses the continuous 6D rotation representation. Phalanges are capsules
and the palm is a smoothed convex hull. Everything is PyTorch, so every quantity is differentiable.

**Objects.** Exact signed distance functions for spheres, boxes, cylinders, tori and their unions.
Surface normals come from autograd.

**Energy.** For a batch of poses $q$:

$$
E(q) = w_c\sum_k \rho\big(d(c_k)\big) + w_a\sum_k\big(1+n^{\text{hand}}_k\!\cdot n^{\text{obj}}_k\big)
     + w_p\sum_j a_j\big(\delta_j+\delta_j^2\big) + w_f\,E_{\text{closure}} + w_s E_{\text{self}} + w_r\lVert\theta-\theta_0\rVert^2
$$

The terms pull contact points $c_k$ (finger pads, palm) onto the surface with a smooth-L1 penalty
$\rho$, face the pads towards the object, push every hand-surface sample $s_j$ out of the object
(its penetration depth $\delta_j=\mathrm{relu}(-d(s_j))$ in centimetres, weighted by the area
$a_j$ of skin closest to the sample), reward force closure, keep fingers out of each other and the
palm, and keep joints near a relaxed posture.

**Differentiable force closure.** Each contact's friction cone is linearised into primitive
wrenches $W$. The grasp resists every disturbance of size $\delta$ along the 12 directions
$\pm\delta e_i$ exactly when those points lie inside $\mathrm{conv}(W)$. diffgrasp computes each
squared distance $\min_{\lambda\in\Delta}\lVert W\lambda-p_i\rVert^2$ with pairwise Frank–Wolfe, and
differentiates it through Danskin's theorem, $\nabla_W = 2(W\lambda^\ast-p_i)\lambda^{\ast\top}$.
No autograd through the inner solver is needed. Contacts that are not touching have their cones
faded by $e^{-(d/3\,\mathrm{mm})^2}$, so the optimiser cannot "close" a grasp with fingers in mid-air.

**Optimiser.** Each seed starts from a pre-shaped opposition posture whose contact centroid sits at
a random point inside the object. All seeds then run together through Adam with cosine step decay
and annealed exploration noise. Seeds are ranked by energy and the best valid grasp is reported.

## Design notes

Each of these was found by isolating one term at a time, or by checking results with something
independent of the optimiser:

1. **The thumb could not oppose.** A 2-DOF thumb base left the thumb pad facing the wrist. Adding
   CMC axial rotation raised the pad-to-target alignment of the pre-grasp from 0.6 to 0.96.
2. **Contact points sat 3% inside the capsules.** A perfect touch therefore implied 0.25 mm of
   penetration, so the two terms could never both reach zero.
3. **A quadratic contact penalty never quite closes.** Its pull shrinks in proportion to the
   remaining gap. The smooth-L1 penalty pulls with almost constant strength until the last
   millimetre or so, and rounds off to zero within about 0.5 mm so that its gradient stays smooth.
4. **Approaching from outside finds one-sided grasps.** Starting with the object inside a
   pre-shaped aperture finds opposition grasps.
5. **The palm was sampled only on its rim.** The optimiser's palm samples were the hull's vertices,
   which all sit on its rounded rim, and capsule rings were up to 9 mm apart. The independent dense
   check showed 28 of 48 power grasps on the can pressing the palm more than 1 mm into it; the
   optimiser's own samples had flagged one. Samples now cover the whole palm (every point within
   3 mm of one) and the capsules in rings 3 mm apart. Each sample is weighted by the area of skin
   closest to it (its Voronoi cell, counted with dense random points). The penetration energy then
   approximates a surface integral, so its strength depends little on how densely or how evenly the
   skin is sampled. Under the strict check, valid power grasps rose from 4–35% to 46–65% (can:
   8% → 58%).
6. **Wrench hull or unit-normal force balance?** Liu et al. score a grasp by ‖Gc‖, the net wrench of
   equal normal forces at the contacts, and accept a small nonzero value, which they relate to the
   friction the grasp needs. diffgrasp's default term tests the friction-cone wrench hull instead,
   which is what the final ε check measures. An ablation with the same seeds, weights and validity
   check ([`examples/ablation_closure.py`](examples/ablation_closure.py)):

   | Object | Grasp | Valid seeds, ‖Gc‖ | Valid seeds, hull | Median ε of its valid seeds, ‖Gc‖ | Median ε of its valid seeds, hull |
   | --- | --- | ---: | ---: | ---: | ---: |
   | ball | precision | 98% | 98% | 0.064 | 0.087 |
   | cube | precision | 50% | 33% | 0.017 | 0.060 |
   | can | precision | 38% | 10% | 0.022 | 0.052 |
   | mug | power | 27% | 46% | 0.066 | 0.064 |

   Neither term wins outright. ‖Gc‖ makes more precision seeds valid, on the cube and the can. The
   ε columns compare each term's own valid grasps, so they say how sturdy a returned grasp is, not
   how well the term does overall: there the hull term's median is 1.4–3.5× higher on precision
   grasps and equal on the mug. On the mug the hull term also makes more seeds valid. The hull term
   loses precision seeds because their contacts stop short. The median contact sits 1.86 mm (cube)
   and 2.26 mm (can) from the surface, against 0.39 and 0.99 mm with ‖Gc‖, so many miss the 2 mm
   cut-off. A sharper fade of the friction cones with distance, now 3 mm, is the next thing to try.

   Does ‖Gc‖ pull three fingers towards 120°? Equal normal forces balance exactly only when all
   three contact normals are 120° apart, so the test is the three pairwise angles. Medians over the
   valid precision seeds, thumb–index / thumb–middle / index–middle:

   | Object | ‖Gc‖ | hull |
   | --- | --- | --- |
   | ball | 152° / 166° / 33° | 154° / 162° / 44° |
   | cube | 154° / 165° / 35° | 167° / 176° / 13° |
   | can | 167° / 169° / 18° | 179° / 178° / 4° |

   With both terms the thumb opposes two neighbouring fingers. No seed of either term, valid or
   not, has all three angles within 20° of 120°.

## Quick start

```bash
git clone https://github.com/Dy07077/diffgrasp && cd diffgrasp
pip install -e ".[dev]"
pytest -q                          # 50 tests, ~15 s
cd examples && python gallery.py   # writes assets/gallery.png and assets/results.json
```

```python
import diffgrasp as dg

hand, mug = dg.HandModel(), dg.make_object("mug")
opt = dg.GraspOptimizer(hand, mug, grasp_type="power", n_seeds=48)
res = opt.run(800)
report = dg.evaluate(hand, mug, res.pose, opt.contacts)
best = max((r for r in report if r["valid"]), key=lambda r: r["epsilon"])
print(best)   # {'epsilon': ..., 'n_contacts': ..., 'penetration_mm': ..., 'valid': True}
```

## Reproducing the numbers

```bash
cd examples
python gallery.py            # assets/gallery.png and assets/results.json, about 5 min
python ablation_closure.py   # assets/ablation.json, about 7 min
python animate.py            # assets/grasp.gif
python contact_map.py        # assets/contact_maps.png
```

Every script uses RNG seed 0, 48 seeds per object, 800 iterations and the default `EnergyWeights`.
The JSON files record these settings, every weight, the software versions and each seed's result.
The tables on this page came from Python 3.12, PyTorch 2.15 (CPU, 8 threads), NumPy 2.5 and
SciPy 1.18, installed with pip, on an Apple-silicon laptop.
Another machine or library version changes the floating-point order and can move individual seeds.
With 48 seeds per object the 95% Wilson intervals are wide, about ±14 points near 50%, so treat
smaller differences between runs as noise.

## Project layout

```text
diffgrasp/
├── hand.py       procedural hand: kinematics, capsule geometry, contact candidates, meshes
├── objects.py    signed distance functions and marching-cubes meshes
├── energy.py     the grasp energy and grasp types (precision, fingertips, power)
├── wrench.py     friction-cone wrenches and the differentiable force-closure surrogate
├── optimize.py   batched optimiser, pre-grasp initialisation, evaluation, dense penetration check
├── quality.py    Ferrari–Canny epsilon metric (scipy ConvexHull)
├── rotation.py   6D rotations, elementary rotations, Rodrigues
└── render.py     matplotlib renderer with contact heat maps
```

## Limitations

- Grasps are checked quasi-statically with force closure. They are not simulated under gravity or
  shaking yet; a physics check, for example with [softfem](https://github.com/Dy07077/softfem), is
  the natural next step.
- The hand is a capsule mannequin, not a scanned model such as MANO. The optimiser only needs
  differentiable contact points and surface samples, so another hand model can be swapped in.
- Contacts are point contacts with a fixed friction coefficient. Soft-finger contacts would add
  torsional friction and make two-finger pinches force-closure.
- Penetration is checked by sampling, densely and in both directions, but not exactly. A feature
  much smaller than the sample spacing (about 3 mm on the hand, 1–2 mm on the object) could still
  slip through.

## References

1. C. Ferrari, J. Canny. *Planning Optimal Grasps.* ICRA 1992.
2. T. Liu, Z. Liu, Z. Jiao, Y. Zhu, S.-C. Zhu. *Synthesizing Diverse and Physically Stable Grasps With Arbitrary Hand Structures Using Differentiable Force Closure Estimator.* IEEE RA-L, 2021.
3. Y. Zhou, C. Barnes, J. Lu, J. Yang, H. Li. *On the Continuity of Rotation Representations in Neural Networks.* CVPR 2019.
4. S. Lacoste-Julien, M. Jaggi. *On the Global Linear Convergence of Frank-Wolfe Optimization Variants.* NeurIPS 2015.
5. J. Romero, D. Tzionas, M. J. Black. *Embodied Hands: Modeling and Capturing Hands and Bodies Together.* SIGGRAPH Asia 2017.

## License

MIT © 2026 Ding Yuan
