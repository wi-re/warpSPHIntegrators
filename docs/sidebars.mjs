/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  wiki: [
    'index',
    'quickstart',
    'tensor-ode',
    'runs',
    'operator-splitting',
    'options',
    'selection',
    'catalog',
    {
      type: 'category',
      label: 'Families',
      items: [
        'families/explicit-rk',
        'families/symplectic',
        'families/relaxed-chebyshev',
        'families/multistep-explicit',
        'families/dirk',
        'families/coupled-rk',
        'families/multistep-implicit',
        'families/imex',
        'families/rosenbrock',
        'families/exponential',
        'families/newmark',
      ],
    },
    'solver',
    'api',
    'development',
  ],
};

export default sidebars;
