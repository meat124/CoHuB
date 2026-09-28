/*
 * Results rendered from data.js: main benchmark table (Table 2), controlled
 * phase-level results (Figure 4), phase-wise results (Tables 11-13), the
 * observation-scope / policy-sharing study (Table 3) and GPT-6 Astra.
 */
(function () {
  "use strict";

  const D = window.COHUB_DATA;
  const H = window.CoHuB;
  const { el, icon, methodLabel, methodText, taskById, methodById, familyById } = H;

  const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
  const finalSuccess = (m, t) => { const p = D.phaseResults[m][t]; return p ? p[2] : null; };
  const TASKS_2 = D.tasks.filter((t) => t.robots === 2).map((t) => t.id);
  const TASKS_3 = D.tasks.filter((t) => t.robots === 3).map((t) => t.id);
  const bandOf = (f) => (f.training === "per task" ? "task" : "multi");
  // Bar kind of phase i: Figure 4's teal / green when the paper labels it
  // (non-)collaborative, otherwise one of three ordinal greens (p1 light -> p3 dark).
  const kindOf = (phase, i) => (phase.collab === null ? "p" + (i + 1) : phase.collab ? "c" : "nc");
  const KIND_LABEL = { nc: "non-collaborative", c: "collaborative" };
  // Averages over tasks for the phase-wise chart (p3 of avg2 equals the Avg. column of Table 2).
  const AVERAGES = {
    avg2: { text: "2-humanoid tasks", tasks: TASKS_2, note: "Mean over the 8 tasks with two humanoids" },
    avg3: { text: "3-humanoid tasks", tasks: TASKS_3, note: "Mean over the 2 tasks with three humanoids" }
  };
  function phaseValues(methodId, id) {
    if (!AVERAGES[id]) return D.phaseResults[methodId][id];
    const rows = AVERAGES[id].tasks.map((t) => D.phaseResults[methodId][t]);
    return rows.some((r) => !r) ? null : [0, 1, 2].map((i) => mean(rows.map((r) => r[i])));
  }

  // Column headers may break at camel-case boundaries (Trash|Collection), never after "Co".
  function wrapLabel(label) {
    const parts = label.split(/(?=[A-Z][a-z])/);
    if (parts[0] === "Co" && parts.length > 1) parts.splice(0, 2, parts[0] + parts[1]);
    return parts.flatMap((part, i) => (i ? [el("wbr"), part] : [part]));
  }

  /* ==========================================================================
     Table 2: main benchmark results (paper-style table, sortable)
     ========================================================================== */
  function renderLeaderboard(root) {
    const cols = [
      ...TASKS_2.map((t) => ({ key: t, label: taskById[t].name, tasks: [t] })),
      { key: "avg2", label: "Avg.", tasks: TASKS_2, avg: true },
      ...TASKS_3.map((t, i) => ({ key: t, label: taskById[t].name, tasks: [t], sep: i === 0 })),
      { key: "avg3", label: "Avg.", tasks: TASKS_3, avg: true }
    ];
    const value = (m, col) => {
      const vals = col.tasks.map((t) => finalSuccess(m, t));
      if (vals.some((v) => v === null)) return null;
      return col.avg ? mean(vals) : vals[0];
    };
    const best = Object.fromEntries(cols.map((col) => [col.key,
      Math.max(...D.methods.map((m) => value(m.id, col)).filter((v) => v !== null))]));
    const fmt = (v, col) => (v === null ? "–" : col.avg ? v.toFixed(1) : String(v));
    const state = { key: null, dir: null };

    const wrap = el("div", { class: "table-scroll" });
    const reset = el("button", { type: "button", class: "link-btn", hidden: true, text: "Reset order" });
    reset.addEventListener("click", () => { state.key = null; state.dir = null; draw(); });
    root.append(
      el("div", { class: "toolbar" }, [el("span", { class: "toolbar__spacer" }), reset]),
      wrap,
      el("p", { class: "note" }, [
        el("span", null, [el("b", { text: "Highlighted" }), " best per column"]),
        el("span", null, [el("i", { class: "sw", style: "background:var(--band-task)" }), "trained per task"]),
        el("span", null, [el("i", { class: "sw", style: "background:var(--band-multi)" }), "trained multi-task"]),
        el("span", { text: "† LatentToM is evaluated only on tasks with two humanoids" })
      ])
    );

    function headerCell(col) {
      const btn = el("button", { type: "button", "data-dir": state.key === col.key ? state.dir : null,
        "aria-label": `Sort by ${col.avg ? (col.tasks.length === 2 ? "3-humanoid average" : "2-humanoid average") : col.label}` },
      wrapLabel(col.label));
      btn.addEventListener("click", () => {
        if (state.key !== col.key) { state.key = col.key; state.dir = "desc"; }
        else if (state.dir === "desc") state.dir = "asc";
        else { state.key = null; state.dir = null; }
        draw();
      });
      return el("th", { scope: "col", class: [col.avg && "is-avg", col.sep && "sep"].filter(Boolean).join(" ") || null,
        "aria-sort": state.key === col.key ? (state.dir === "desc" ? "descending" : "ascending") : null }, btn);
    }

    function methodRow(m, showFamily) {
      const th = el("th", { class: "lb__method", scope: "row" }, [methodLabel(m), m.note ? el("sup", { title: m.note, text: "†" }) : null,
        showFamily ? el("span", { class: "fam-tag", text: familyById[m.family].name }) : null]);
      return el("tr", null, [th, ...cols.map((col, ci) => {
        const v = value(m.id, col);
        const isBest = v !== null && v === best[col.key];
        return el("td", {
          "data-col": String(ci),
          class: [isBest && "is-best", v === null && "is-na", col.avg && "is-avg", col.sep && "sep"].filter(Boolean).join(" ") || null,
          title: v === null ? "Not evaluated" : null
        }, isBest ? el("b", { text: fmt(v, col) }) : fmt(v, col));
      })]);
    }

    function draw() {
      const thead = el("thead", null, [
        el("tr", { class: "lb__groups" }, [
          el("th", { class: "lb__method" }),
          el("th", { colspan: String(TASKS_2.length + 1), scope: "colgroup", class: "lb__grouped", text: "2 Humanoids" }),
          el("th", { colspan: String(TASKS_3.length + 1), scope: "colgroup", class: "lb__grouped sep", text: "3 Humanoids" })
        ]),
        el("tr", { class: "lb__cols" }, [el("th", { class: "lb__method", scope: "col", text: "Method" }), ...cols.map(headerCell)])
      ]);

      const tbody = el("tbody");
      if (state.key) {
        const col = cols.find((c) => c.key === state.key);
        D.methods.slice().sort((a, b) => {
          const va = value(a.id, col), vb = value(b.id, col);
          if (va === null) return 1;
          if (vb === null) return -1;
          return state.dir === "desc" ? vb - va : va - vb;
        }).forEach((m) => tbody.append(methodRow(m, true)));
      } else {
        for (const f of D.families) {
          tbody.append(el("tr", { class: "lb__family lb__family--" + bandOf(f) },
            el("th", { colspan: String(cols.length + 1), scope: "rowgroup" }, el("span", { text: f.long }))));
          D.methods.filter((m) => m.family === f.id).forEach((m) => tbody.append(methodRow(m, false)));
        }
      }

      // Task average across evaluated policies, computed from the rows above.
      const tfoot = el("tfoot", null, el("tr", null, [
        el("th", { class: "lb__method", scope: "row", text: "Task avg." }),
        ...cols.map((col) => {
          if (col.avg) return el("td", { class: "is-avg" });
          const vals = D.methods.map((m) => finalSuccess(m.id, col.key)).filter((v) => v !== null);
          return el("td", { class: col.sep ? "sep" : null, text: mean(vals).toFixed(1) });
        })
      ]));

      const table = el("table", { class: "lb" }, [
        el("caption", { class: "sr-only", text: "Final task success rate (%) per method and task" }), thead, tbody, tfoot]);
      // Column highlight follows the pointer.
      table.addEventListener("pointerover", (e) => {
        const cell = e.target.closest("td[data-col]");
        table.querySelectorAll("td.is-col").forEach((c) => c.classList.remove("is-col"));
        if (cell) table.querySelectorAll(`tbody td[data-col="${cell.dataset.col}"]`).forEach((c) => c.classList.add("is-col"));
      });
      table.addEventListener("pointerleave", () => table.querySelectorAll("td.is-col").forEach((c) => c.classList.remove("is-col")));

      reset.hidden = !state.key;
      wrap.replaceChildren(table);
    }
    draw();
  }

  /* ==========================================================================
     Bar chart: methods grouped by family on x, success (%) on y.
     Plain HTML/CSS so bars animate with CSS transitions.
     ========================================================================== */
  function createBarChart({ barsPerGroup, subs = false, compact = false, yTitle = "", ariaLabel }) {
    const ticks = [0, 20, 40, 60, 80, 100];
    const root = el("div", { class: "bc is-pending" + (compact ? " bc--compact" : "") + (barsPerGroup === 3 ? " bc--three" : ""), role: "group", "aria-label": ariaLabel });
    root.style.setProperty("--bar-w", barsPerGroup === 2 ? "32%" : "25%");
    const legend = el("div", { class: "bc__legend" });
    const families = el("div", { class: "bc__families" });
    const refs = {};
    let gi = 0;
    D.families.forEach((f, fi) => {
      const members = D.methods.filter((m) => m.family === f.id);
      const groups = el("div", { class: "bc__groups" });
      for (const m of members) {
        const area = el("div", { class: "bc__area" });
        const bars = [];
        for (let i = 0; i < barsPerGroup; i++) {
          const val = el("span", { class: "bc__val" });
          const bar = el("div", { class: "bc__bar", style: `--i:${i}` }, val);
          area.append(bar);
          bars.push({ bar, val });
        }
        const na = el("div", { class: "bc__na", text: "not evaluated", hidden: true });
        area.append(na);
        const top = el("div", { class: "bc__top" });
        const subRow = subs ? el("div", { class: "bc__subs", "aria-hidden": "true" },
          bars.map((_, i) => el("span", { class: "bc__sub", text: `p${i + 1}` }))) : null;
        const group = el("div", { class: "bc__group", tabindex: "0", style: `--gi:${gi++}` },
          [top, area, subRow, el("div", { class: "bc__label" }, methodLabel(m))]);
        groups.append(group);
        refs[m.id] = { group, bars, top, na, m };
      }
      // The dashed rule separates per-task families from multi-task ones, as in Figure 4.
      families.append(el("div", { class: "bc__family" + (fi > 0 && bandOf(f) !== bandOf(D.families[fi - 1]) ? " bc__family--split" : ""),
        style: `--n:${members.length}` }, [groups, el("div", { class: "bc__family-label", text: f.name })]));
    });
    root.append(legend, el("div", { class: "bc__scroll" }, el("div", { class: "bc__body" }, [
      el("div", { class: "bc__ytitle", "aria-hidden": "true" }, el("span", { text: yTitle })),
      el("div", { class: "bc__yaxis", "aria-hidden": "true" }, ticks.map((t) => el("span", { style: `--y:${t}`, text: String(t) }))),
      el("div", { class: "bc__plot" }, [el("div", { class: "bc__grid", "aria-hidden": "true" }, ticks.map((t) => el("span", { style: `--y:${t}` }))), families])
    ])));

    // Hover or focus a method: highlight it, dim the others, show the numbers.
    let tipFor = null;
    for (const ref of Object.values(refs)) {
      const show = (x, y) => {
        root.classList.add("has-hot");
        ref.group.classList.add("is-hot");
        if (tipFor) H.tooltip.show(tipFor(ref.m.id), x, y);
      };
      const hide = () => {
        root.classList.remove("has-hot");
        ref.group.classList.remove("is-hot");
        H.tooltip.hide();
      };
      ref.group.addEventListener("pointermove", (e) => show(e.clientX, e.clientY));
      ref.group.addEventListener("pointerleave", hide);
      ref.group.addEventListener("focus", () => { const b = ref.group.getBoundingClientRect(); show(b.right - 8, b.top + 40); });
      ref.group.addEventListener("blur", hide);
    }

    // Bars grow the first time the chart scrolls into view.
    if (H.reduceMotion.matches) root.classList.remove("is-pending");
    else H.onceVisible([root], () => requestAnimationFrame(() => root.classList.remove("is-pending")), { threshold: 0.2 });

    return { root, legend, refs, setTip(fn) { tipFor = fn; } };
  }

  function tipRow(color, value, label) {
    return el("div", { class: "tt-row" }, [el("span", { class: "tt-key", style: `background:${color}` }), el("strong", { text: value }), el("span", { class: "tt-muted", text: label })]);
  }
  // Bar colours of the paper's Figure 4 (also used for tooltip keys, which live outside the chart).
  const KIND_COLOR = { nc: "#79b1cd", c: "#b2d4a6", p1: "#c6e8d1", p2: "#8cc9a3", p3: "#3f9a61" };
  const swatch = (kind) => el("span", { class: "bc__sw", "data-kind": kind, "aria-hidden": "true" });

  function legendKey(root, kind, label) {
    const key = el("button", { type: "button", class: "bc__key", "aria-pressed": "true" }, [swatch(kind), label]);
    key.addEventListener("click", () => {
      const off = key.getAttribute("aria-pressed") === "true";
      key.setAttribute("aria-pressed", String(!off));
      root.toggleAttribute("data-off-" + kind, off);
    });
    return key;
  }

  /* ==========================================================================
     Figure 4: controlled phase-level results (non-collaborative vs collaborative)
     ========================================================================== */
  function renderGap(root) {
    const chart = createBarChart({ barsPerGroup: 2, yTitle: "Phase Success Rate (%)",
      ariaLabel: "Average success on non-collaborative and collaborative phases for eight policies" });
    const maxDrop = Math.min(...D.collabGap.map((r) => r.drop));
    for (const r of D.collabGap) {
      const ref = chart.refs[r.method];
      const [nc, c] = ref.bars;
      nc.bar.dataset.kind = "nc";
      nc.bar.style.setProperty("--v", r.nc);
      nc.val.textContent = r.nc.toFixed(1);
      c.bar.dataset.kind = "c";
      c.bar.style.setProperty("--v", r.c);
      c.val.textContent = r.c.toFixed(1);
      ref.top.textContent = "−" + Math.abs(r.drop).toFixed(1) + "%";
      ref.top.classList.toggle("is-max", r.drop === maxDrop);
      ref.group.setAttribute("aria-label", `${methodText(ref.m)}: non-collaborative ${r.nc}%, collaborative ${r.c}%, relative drop ${Math.abs(r.drop)}%`);
    }
    chart.setTip((id) => {
      const r = D.collabGap.find((x) => x.method === id);
      return el("div", null, [
        el("div", { class: "tt-title" }, methodLabel(methodById[id])),
        tipRow(KIND_COLOR.nc, r.nc.toFixed(1) + "%", "non-collaborative"),
        tipRow(KIND_COLOR.c, r.c.toFixed(1) + "%", "collaborative"),
        el("div", { class: "tt-muted", text: `Relative drop ${Math.abs(r.drop).toFixed(1)}%` })
      ]);
    });
    chart.legend.append(
      legendKey(chart.root, "nc", "Non-collaborative (NC)"),
      legendKey(chart.root, "c", "Collaborative (C)"),
      el("span", { class: "bc__note" }, [el("b", { text: "-x%" }), "Relative drop (C-NC)/NC"])
    );

    // Accessible twin of the chart.
    const tableBtn = el("button", { type: "button", class: "link-btn", "aria-expanded": "false" }, [icon("table"), "Table"]);
    const table = el("div", { class: "table-scroll", hidden: true, style: "margin-top:18px" }, el("table", { class: "lb lb--compact" }, [
      el("thead", null, el("tr", { class: "lb__cols" }, ["Method", "Non-collaborative", "Collaborative", "Relative drop"]
        .map((h, i) => el("th", { scope: "col", class: i === 0 ? "lb__left" : null, text: h })))),
      el("tbody", null, D.collabGap.map((r) => el("tr", null, [
        el("th", { scope: "row", class: "lb__left" }, methodLabel(methodById[r.method])),
        el("td", { text: r.nc.toFixed(1) }), el("td", { text: r.c.toFixed(1) }), el("td", { text: r.drop.toFixed(1) + "%" })
      ])))
    ]));
    tableBtn.addEventListener("click", () => {
      table.hidden = !table.hidden;
      tableBtn.setAttribute("aria-expanded", String(!table.hidden));
    });
    root.append(el("div", { class: "toolbar" }, [el("span", { class: "toolbar__spacer" }), tableBtn]), chart.root, table);
  }

  /* ==========================================================================
     Tables 11-13: phase-wise results, one task at a time
     ========================================================================== */
  function phaseSequence(names, kinds) {
    const out = [];
    names.forEach((name, i) => {
      if (i) out.push(el("span", { class: "bc__arrow", "aria-hidden": "true" }, icon("arrow")));
      out.push(el("span", { class: "bc__phase" }, [swatch(kinds[i]), el("b", { text: `p${i + 1}` }), name,
        KIND_LABEL[kinds[i]] ? el("span", { class: "sr-only", text: ` (${KIND_LABEL[kinds[i]]})` }) : null]));
    });
    return el("span", { class: "bc__phases" }, out);
  }

  // `id` is a task id, or "avg2" / "avg3" for the mean over the two- or three-humanoid tasks.
  function fillPhaseChart(chart, id) {
    const avg = AVERAGES[id];
    const t = avg ? null : taskById[id];
    const kinds = t ? t.phases.map(kindOf) : ["p1", "p2", "p3"];
    const names = t ? t.phases.map((p) => p.name) : ["", "", "full-task success"];
    const title = avg ? `Average over ${avg.text}` : t.name;
    const fmt = (v) => (avg ? v.toFixed(1) : String(v));

    chart.root.removeAttribute("data-off-nc");
    chart.root.removeAttribute("data-off-c");
    const legend = [];
    if (kinds.includes("nc")) legend.push(legendKey(chart.root, "nc", "Non-collaborative"));
    if (kinds.includes("c")) legend.push(legendKey(chart.root, "c", "Collaborative"));
    legend.push(phaseSequence(names, kinds));
    if (t && t.phaseNote) legend.push(el("span", { class: "bc__caption", text: t.phaseNote }));
    if (avg) legend.push(el("span", { class: "bc__caption", text: `${avg.note}; phases are not split into collaborative / non-collaborative.` }));
    chart.legend.replaceChildren(...legend);

    for (const ref of Object.values(chart.refs)) {
      const vals = phaseValues(ref.m.id, id);
      ref.na.hidden = Boolean(vals);
      ref.bars.forEach(({ bar, val }, i) => {
        bar.dataset.kind = kinds[i];
        bar.style.setProperty("--v", vals ? vals[i] : 0);
        bar.toggleAttribute("data-zero", !vals || vals[i] === 0);
        val.textContent = vals ? fmt(vals[i]) : "";
      });
      ref.group.setAttribute("aria-label", vals
        ? `${methodText(ref.m)}, ${title}: ` + vals.map((v, i) => `p${i + 1} ${fmt(v)}%`).join(", ")
        : `${methodText(ref.m)}: not evaluated (${title})`);
    }
    chart.setTip((methodId) => {
      const vals = phaseValues(methodId, id);
      const head = el("div", { class: "tt-title" }, methodLabel(methodById[methodId]));
      if (!vals) return el("div", null, [head, el("div", { class: "tt-muted", text: "Not evaluated" })]);
      return el("div", null, [head,
        ...vals.map((v, i) => tipRow(KIND_COLOR[kinds[i]], fmt(v) + "%", `p${i + 1}${names[i] ? " " + names[i] : ""}`)),
        avg ? el("div", { class: "tt-muted", text: avg.note }) : null]);
    });
  }

  function phaseChart(taskId, compact) {
    const chart = createBarChart({ barsPerGroup: 3, subs: true, compact, yTitle: "Cumulative Success (%)",
      ariaLabel: `Cumulative phase success on ${taskById[taskId].name}` });
    fillPhaseChart(chart, taskId);
    return chart.root;
  }

  function renderPhaseChart(root) {
    const chart = createBarChart({ barsPerGroup: 3, subs: true, yTitle: "Cumulative Success (%)", ariaLabel: "Cumulative phase success per method" });
    const tabs = H.taskTabs((id) => show(id), {
      extra: { label: "Average", items: Object.entries(AVERAGES).map(([id, a]) => ({ id, text: a.text })) }
    });
    function show(id) {
      tabs.select(id);
      fillPhaseChart(chart, id);
    }
    root.append(tabs.root, chart.root);
    show("avg2");
  }

  /* ==========================================================================
     Table 3: observation scope x policy sharing
     ========================================================================== */
  function renderDesignTable(root) {
    const { models, rows } = D.designChoices;
    const best = models.map((_, i) => Math.max(...rows.map((r) => r.values[i])));
    root.append(el("div", { class: "table-scroll" }, el("table", { class: "lb lb--compact" }, [
      el("caption", { class: "sr-only", text: "Average task success (%) over the eight two-humanoid tasks by observation scope and policy sharing" }),
      el("thead", null, [
        el("tr", { class: "lb__groups" }, [
          el("th", { colspan: "2" }),
          el("th", { colspan: "2", scope: "colgroup", class: "lb__grouped", text: "Standard IL" }),
          el("th", { scope: "colgroup", class: "lb__grouped sep", text: "VLA" })
        ]),
        el("tr", { class: "lb__cols" }, [
          el("th", { scope: "col", class: "lb__left", text: "Obs. scope" }), el("th", { scope: "col", class: "lb__left", text: "Policy" }),
          ...models.map((id, i) => el("th", { scope: "col", class: i === 2 ? "sep" : null }, methodLabel(methodById[id])))
        ])
      ]),
      el("tbody", null, rows.map((r) => el("tr", null, [
        el("th", { scope: "row", class: "lb__left", text: r.scope }), el("td", { class: "lb__left", text: r.policy }),
        ...r.values.map((v, i) => el("td", { class: [v === best[i] && "is-best", i === 2 && "sep"].filter(Boolean).join(" ") || null },
          v === best[i] ? el("b", { text: v.toFixed(1) }) : v.toFixed(1)))
      ])))
    ])));
  }

  /* ==========================================================================
     GPT-6 Astra vs the best trained policy on each two-humanoid task
     ========================================================================== */
  function renderAstra(root) {
    if (!root || !D.astra) return;
    const rows = TASKS_2.map((t) => {
      let best = null;
      for (const m of D.methods) {
        const v = finalSuccess(m.id, t);
        if (v !== null && (!best || v > best.v)) best = { v, m };
      }
      return { t, astra: D.astra.phases[t][2], best };
    });
    const bar = (kind, v, who) => el("div", { class: "ab__bar", "data-kind": kind, "data-zero": v === 0 ? "" : null, style: `--v:${v}` },
      el("span", { class: "ab__val" }, [String(v), who ? el("span", { class: "ab__who" }, who) : null]));
    const chart = el("div", { class: "bc ab is-pending" }, [
      el("div", { class: "bc__legend" }, [
        el("span", { class: "bc__key" }, [swatch("astra"), `${D.astra.name} (${D.astra.rollouts} episodes)`]),
        el("span", { class: "bc__key" }, [swatch("best"), "Best trained policy (100 rollouts)"])
      ]),
      el("div", { class: "ab__rows" }, rows.map((r) => el("div", {
        class: "ab__row" + (r.astra > r.best.v ? " is-win" : ""),
        "aria-label": `${taskById[r.t].name}: ${D.astra.name} ${r.astra}%, best trained policy ${methodText(r.best.m)} ${r.best.v}%`
      }, [
        el("span", { class: "ab__task", text: taskById[r.t].name }),
        el("div", { class: "ab__bars", "aria-hidden": "true" }, [bar("astra", r.astra), bar("best", r.best.v, methodLabel(r.best.m))])
      ]))),
      el("div", { class: "ab__axis", "aria-hidden": "true" }, [0, 20, 40, 60, 80, 100].map((x) => el("span", { style: `--x:${x}`, text: String(x) }))),
      el("p", { class: "ab__title", text: "Task success (%)" })
    ]);
    root.append(chart);
    H.onceVisible([chart], () => chart.classList.remove("is-pending"), { threshold: 0.2 });
  }

  H.phaseChart = phaseChart;

  document.addEventListener("DOMContentLoaded", () => {
    renderLeaderboard(H.$("[data-leaderboard]"));
    renderGap(H.$("[data-gap-chart]"));
    renderPhaseChart(H.$("[data-phase-chart]"));
    renderDesignTable(H.$("[data-design-table]"));
    renderAstra(H.$("[data-astra-chart]"));
  });
})();
