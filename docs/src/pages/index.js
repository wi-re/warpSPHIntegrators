import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import FeatureCard from '@site/src/components/FeatureCard';

export default () => (
  <Layout
    title="warpSPHIntegrators — scheme wiki"
    description="Differentiable ODE integrators for PyTorch: 137 schemes across 11 families, matrix-free implicit machinery, end-to-end autograd."
  >
    <header className="hero hero--primary">
      <div className="container text--center">
        <h1>
          warpSPHIntegrators
          <span className="heroSubtitle">scheme wiki</span>
        </h1>
        <p className="heroTagline">
          Differentiable ODE integrators for PyTorch — 137 schemes across 11
          families, from explicit RK through coupled fully-implicit
          collocation to exponential integrators.
        </p>
        <div className="row">
          <div className="col">
            <Link
              className="button button--primary button--lg"
              to="docs/families/explicit-rk"
            >
              Browse the wiki →
            </Link>
          </div>
          <div className="col">
            <Link
              className="button button--secondary button--lg"
              to="https://github.com/wi-re/warpSPHIntegrators"
            >
              GitHub
            </Link>
          </div>
        </div>
      </div>
    </header>

    <main>
      <div className="container pad-top--lg pad-bottom--lg">
        <div className="row">
          <FeatureCard title="137 schemes, 11 families">
            Equations and Butcher tableaus with source citations and documented
            corrections, stability, and per-step cost — with the
            test files that pin every measured claim.
          </FeatureCard>
          <FeatureCard title="Matrix-free implicit machinery">
            JFNK + GMRES with exact-JVP matvecs, left/right
            preconditioning, and the work-unit invariant that makes solver
            overhead benchmarkable.
          </FeatureCard>
          <FeatureCard title="Differentiable end-to-end">
            Every implicit scheme is differentiable through the solve via
            implicit-function-theorem re-attachment — exact on linear
            problems.
          </FeatureCard>
        </div>
      </div>
    </main>
  </Layout>
);
