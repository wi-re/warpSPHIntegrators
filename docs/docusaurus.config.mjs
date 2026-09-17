import { fileURLToPath } from 'node:url';
import { themes as prismThemes } from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'warpSPHIntegrators',
  tagline:
    'Differentiable ODE integrators for PyTorch — 64 schemes across 11 families',
  favicon: 'img/favicon.svg',
  organizationName: 'wi-re',
  projectName: 'warpSPHIntegrators',

  // The account's custom domain serves the repo under this base path.
  url: 'https://fluids.dev',
  baseUrl: '/warpSPHIntegrators/',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',
  onBrokenAnchors: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  themes: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        indexDocs: true,
        indexPages: true,
        indexBlog: false,
        docsRouteBasePath: '/docs',
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: fileURLToPath(new URL('./sidebars.mjs', import.meta.url)),
          sidebarCollapsed: false,
          // Math rendering (KaTeX). remark-math registers a micromark
          // extension, so $...$ / $$...$$ spans are tokenized before MDX's
          // expression parser sees the LaTeX braces inside them.
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          editUrl: ({ docPath }) =>
            `https://github.com/wi-re/warpSPHIntegrators/edit/main/docs/docs/${docPath}`,
        },
        blog: false,
        theme: {
          customCss: fileURLToPath(new URL('./src/css/custom.css', import.meta.url)),
        },
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'light',
      },

      navbar: {
        title: 'warpSPHIntegrators',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'wiki',
            position: 'left',
            label: 'Families',
          },
          { to: 'docs/solver', position: 'left', label: 'Solver stack' },
          {
            to: 'https://github.com/wi-re/warpSPHIntegrators',
            label: 'GitHub',
            position: 'right',
            className: 'header-github-link',
          },
        ],
      },

      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              { label: 'Wiki', to: 'docs' },
              { label: 'Nonlinear solver stack', to: 'docs/solver' },
              { label: 'Explicit RK', to: 'docs/families/explicit-rk' },
              { label: 'DIRK', to: 'docs/families/dirk' },
              { label: 'Coupled fully implicit RK', to: 'docs/families/coupled-rk' },
            ],
          },
          {
            title: 'Repository',
            items: [
              {
                label: 'GitHub',
                to: 'https://github.com/wi-re/warpSPHIntegrators',
              },
              {
                label: 'README',
                to: 'https://github.com/wi-re/warpSPHIntegrators/blob/main/README.md',
              },
              {
                label: 'NOTES (measurements)',
                to: 'https://github.com/wi-re/warpSPHIntegrators/blob/main/NOTES.md',
              },
              {
                label: 'Roadmap',
                to: 'https://github.com/wi-re/warpSPHIntegrators/blob/main/IMPLICIT_ROADMAP.md',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Rene Winchenbach. Built with Docusaurus.`,
      },

      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.githubDark,
      },

      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3,
      },
    }),
};

export default config;
