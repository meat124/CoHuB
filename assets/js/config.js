/*
 * Site settings: venue, authors, links and BibTeX.
 */
window.COHUB_CONFIG = {
  // Pill above the title. An empty string hides it.
  venue: "Under review",

  // Each inner list is one line of the author block.
  // Optional fields: url (personal page), notes (keys of authorNotes below).
  authors: [
    [
      { name: "Hyunjin Park", affiliations: [1], notes: ["equal", "lead"], url: "https://meat124.github.io/" },
      { name: "Jebeom Chae", affiliations: [1], notes: ["equal"], url: "https://jebeom.github.io/" },
      { name: "Minwoo Park", affiliations: [1], notes: ["equal"], url: "https://minwoopark00.github.io/" }
    ],
    [
      { name: "Sunghyun Park", affiliations: [1], url: "https://edipark.github.io/" },
      { name: "Hanjun Yoo", affiliations: [2], url: "https://lukehanjun.github.io/Portfolio/" },
      { name: "Seoyeon Choi", affiliations: [3], url: "https://n00nspr1ng.github.io/" },
      { name: "Soochul Yoo", affiliations: [1] },
      { name: "Joohwan Seo", affiliations: [3], url: "https://sites.google.com/berkeley.edu/joohwan-seo-page/home" },
      { name: "Sarmad Idrees", affiliations: [1] }
    ],
    [
      { name: "Jae-Sang Hyun", affiliations: [1], url: "https://scholar.google.com/citations?hl=en&user=5Es881oAAAAJ" },
      { name: "Jongmin Lee", affiliations: [1], url: "https://www.jmlee.kr/" },
      { name: "Roberto Horowitz", affiliations: [3], url: "https://me.berkeley.edu/people/roberto-horowitz/" },
      { name: "Youngwoon Lee", affiliations: [4], url: "https://youngwoon.github.io/" },
      { name: "Jongeun Choi", affiliations: [1], notes: ["corresponding"], url: "https://mlcs.yonsei.ac.kr/people" }
    ]
  ],
  // Numbered from 1 in the order given.
  affiliations: [
    "Yonsei University",
    "Gwangju Institute of Science and Technology",
    "University of California, Berkeley",
    "Seoul National University"
  ],
  // Footnote marks for `notes`. Each one is listed under the affiliations, in
  // this order, when at least one author uses it.
  authorNotes: {
    equal: { mark: "*", text: "Co-first authors" },
    lead: { mark: "†", text: "Project lead" },
    corresponding: { mark: "‡", text: "Corresponding author" }
  },

  // Empty string = not available yet. Missing links render as a disabled
  // "soon" button (or are hidden when showComingSoon is false).
  // `where`: "title" = buttons under the title (and top-right of the hero when
  // available), "data" = buttons in the dataset section.
  links: [
    { id: "paper", label: "Paper", icon: "paper", url: "", where: "title" },
    { id: "arxiv", label: "arXiv", icon: "arxiv", url: "", where: "title" },
    { id: "code", label: "Code", icon: "code", url: "", where: "title" },
    { id: "dataset", label: "Dataset", icon: "data", url: "", where: "title" },
    { id: "data2", label: "Dataset · 2 humanoids", icon: "data", url: "", where: "data" },
    { id: "data3", label: "Dataset · 3 humanoids", icon: "data", url: "", where: "data" }
  ],
  showComingSoon: true,

  // Empty until the paper is on arXiv: the BibTeX button and the Citation
  // section then show "soon".
  bibtex: "",

  // Optional acknowledgements paragraph.
  acknowledgements: ""
};
