/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  wiki: [
    'index',
    'quickstart',
    'selection',
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
  ],
};

export default sidebars;
