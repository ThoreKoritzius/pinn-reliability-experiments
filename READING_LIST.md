# Reading path: from Taylor–Green to PINN reliability

Compiled 16 November 2026. Start coding alongside the first three readings;
you do not need to finish the theoretical papers before running the toy experiment.
Times below are rough first-pass estimates, not estimates for mastering each paper.

## First: enough physics and PINNs to understand our experiment

1. **Nektar++: Taylor–Green vortex tutorial** — [tutorial PDF](https://doc.nektar.info/tutorials/latest/incns/taylor-green-vortex/incns-taylor-green-vortex.pdf).
   Spend 30–60 minutes. Read the two-dimensional exact solution and the setup of an
   incompressible-flow verification problem. Work out divergence, viscous decay and
   the pressure–advection cancellation yourself. Authors use different sine/cosine
   conventions: compare equations rather than copying a pressure sign.
   **Question:** why can we test numerical accuracy here without a CFD reference solver?

2. **Raissi, Perdikaris & Karniadakis (2019), Physics-informed neural networks:
   A deep learning framework for solving forward and inverse problems involving
   nonlinear partial differential equations** — [published paper](https://doi.org/10.1016/j.jcp.2018.10.045),
   [authors' early Part I preprint](https://arxiv.org/abs/1711.10561).
   Spend 1–2 hours on the forward-problem formulation, automatic differentiation,
   and construction of the loss. The Part I preprint is a precursor, not an
   identical copy of the journal article.
   **Question:** what information constrains a PINN when there are no interior labels?

3. **Wang, Sankaran, Wang & Perdikaris (2023 preprint), An Expert's Guide to Training
   Physics-informed Neural Networks** — [paper](https://arxiv.org/abs/2308.08468).
   Spend 1–2 hours first on scaling, architecture, optimization and loss balancing.
   Return to advanced techniques when a baseline fails. This is the practical
   companion to the code, not a recipe to adopt every improvement simultaneously.
   **Question:** which changes alter the hypothesis class, the optimization, or the sampling?

## Next: understand why a small loss can coexist with an inaccurate solution

4. **Krishnapriyan et al. (NeurIPS 2021), Characterizing possible failure modes in
   physics-informed neural networks** — [paper](https://arxiv.org/abs/2109.01050).
   Spend 60–90 minutes on the experiments and curriculum discussion.
   **Question:** can an easy PDE become difficult for this optimization formulation,
   and what evidence distinguishes that from insufficient network size?

5. **Wang, Teng & Perdikaris (SIAM J. Scientific Computing, 2021), Understanding and
   Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks** —
   [journal](https://doi.org/10.1137/20M1318043), [preprint](https://arxiv.org/abs/2001.04536).
   Spend 1–2 hours on gradient imbalance and the composite loss.
   **Question:** why does multiplying the initial-condition penalty by ten change
   training even though the exact solution minimizes both formulations?

6. **De Ryck & Mishra (Acta Numerica, 2024), Numerical analysis of physics-informed
   neural networks and related models in physics-informed machine learning** —
   [open preprint](https://arxiv.org/abs/2402.10926),
   [journal](https://doi.org/10.1017/S0962492923000089).
   Skim the framework first; revisit selected sections over several sessions.
   Focus on approximation, training, generalization, and PDE stability.
   **Question:** which link is missing between empirical residual loss and solution error?

7. **De Ryck, Jagtap & Mishra (IMA J. Numerical Analysis, 2024), Error estimates for
   physics-informed neural networks approximating the Navier–Stokes equations** —
   [open preprint](https://arxiv.org/abs/2203.09346),
   [journal](https://academic.oup.com/imajna/article/44/1/83/6989836).
   Read the setup and main theorem statements before attempting proofs. Trace the
   assumptions, residual/boundary terms and stability constants.
   **Question:** is a bound computable and informative for our trained network, or
   does it mainly explain a qualitative relationship?

## Then: the research neighborhood of “does it know it is wrong?”

8. **Hillebrecht & Unger (2025 preprint), Prediction error certification for PINNs:
   Theory, computation, and application to Stokes flow** — [paper](https://arxiv.org/abs/2508.07994).
   Focus on required stability quantities and how they are approximated. Stokes
   flow omits nonlinear advection; distinguish it from general Navier–Stokes.
   **Question:** what does certification cost, and which assumptions make it possible?

9. **Gopakumar et al. (ICML 2025), Calibrated Physics-Informed Uncertainty
   Quantification** — [paper](https://arxiv.org/abs/2502.04406),
   [full text](https://arxiv.org/html/2502.04406v2).
   Read the score construction and limitations together. The authors explicitly
   distinguish residual-space coverage from physical-variable coverage.
   **Question:** what exactly is covered by the prediction set?

10. **Yu, Ho & Wang (2025 preprint), A Conformal Prediction Framework for Uncertainty
    Quantification in Physics-Informed Neural Networks** — [paper](https://arxiv.org/abs/2509.13717).
    Compare calibration with solution labels against residual-based calibration.
    **Question:** what calibration information is required, and what happens if test
    problems are not distributed like calibration problems?

11. **Hao et al. (NeurIPS 2024), PINNacle: A Comprehensive Benchmark of
    Physics-Informed Neural Networks for Solving PDEs** —
    [paper](https://papers.nips.cc/paper_files/paper/2024/file/8c63299fb2820ef41cb05e2ff11836f5-Paper-Datasets_and_Benchmarks_Track.pdf),
    [official code](https://github.com/i207M/PINNacle).
    Skim the problem suite and experimental protocol once we have our toy result.
    **Question:** what would a reliability benchmark add beyond an accuracy leaderboard?

## Suggested sequence

- **First weekend:** 1–3; derive the solution and run `taylor_green.py --check`.
- **While training and inspecting errors:** 4–5; compare Adam and L-BFGS checkpoints.
- **After our first experiment:** 6–8; connect diagnostics to mathematical error estimates.
- **Before choosing a research claim:** 9–11; build a matrix of inputs, guarantees,
  evaluation splits, compute and failure cases for the nearest methods.

For each paper keep four notes: **claim, assumptions, evidence, failure/limitation**.
Keep theoretical guarantees, empirical error detection, and physical model validity
separate. This list is a learning path, not an exhaustive related-work review.
