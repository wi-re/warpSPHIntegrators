import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import FeatureCard from '@site/src/components/FeatureCard';
import catalog from '@site/src/data/catalog.json';

export default () => (
  <Layout
    title="warpSPHIntegrators — methods and examples"
    description={`Differentiable ODE integrators for PyTorch: ${catalog.summary.scheme_count} schemes across ${catalog.summary.family_count} families, matrix-free implicit machinery, end-to-end autograd.`}
  >
    <header className="hero hero--primary">
      <div className="container text--center">
        <h1>
          warpSPHIntegrators
          <span className="heroSubtitle">Methods, examples and practical guidance</span>
        </h1>
        <p className="heroTagline">
          Differentiable ODE integrators for PyTorch — {catalog.summary.scheme_count} schemes across {catalog.summary.family_count} families,
          from explicit RK through coupled fully-implicit
          collocation to exponential integrators.
        </p>
        <div className="row">
          <div className="col">
            <Link
              className="button button--primary button--lg"
              to="docs/quickstart"
            >
              Run your first example →
            </Link>
          </div>
          <div className="col">
            <Link
              className="button button--secondary button--lg"
              to="docs/selection"
            >
              Choose a method
            </Link>
          </div>
        </div>
      </div>
    </header>

    <main>
      <div className="container pad-top--lg pad-bottom--lg">
        <div className="row">
          <FeatureCard title={`${catalog.summary.scheme_count} schemes, ${catalog.summary.family_count} families`}>
            Start with decay and a damped oscillator, then compare accuracy,
            stability and measured work. Friendly method names appear alongside
            their exact enum values and coefficient references.
          </FeatureCard>
          <FeatureCard title="Solving stiff problems">
            Implicit methods solve stage equations. Jacobian-free Newton–Krylov
            (JFNK) uses Jacobian actions rather than a stored matrix; diagnostics
            report nonlinear convergence, RHS work and Krylov iterations.
          </FeatureCard>
          <FeatureCard title="Gradients and custom systems">
            Begin with a tensor ODE or use the state protocol for coupled fields
            and stage-dependent quantities. Each family documents its gradient,
            operator and history assumptions, with runnable examples.
          </FeatureCard>
        </div>
      </div>
    </main>
  </Layout>
);
