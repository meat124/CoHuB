/*
 * Task suite: taxonomy with crossing axes (Figure 2), task detail dialog,
 * egocentric viewer, phase flow (evaluation section) and the failure carousel.
 */
(function () {
  "use strict";

  const D = window.COHUB_DATA;
  const H = window.CoHuB;
  const { $, el, icon, taskById, makeVideo, taskVideoSrc, videos } = H;
  const LEVEL = { low: "Low", high: "High" };
  const ARROW = { low: "↓", high: "↑" };
  const posterOf = (id) => `assets/img/posters/${id}.webp`;
  const kindOf = (phase) => (phase.collab === null ? "n" : phase.collab ? "c" : "nc");

  /* ---------- Taxonomy: 2 x 2 quadrants, axes crossing in the middle ---------- */
  function taskCard(t) {
    const card = el("button", { type: "button", class: "task-card", "data-task": t.id, "aria-label": `${t.name}: ${t.summary} Open task details.` }, [
      el("div", { class: "task-card__media" }, makeVideo({ src: taskVideoSrc(t.id, false), poster: posterOf(t.id) })),
      el("div", { class: "task-card__body" }, [
        el("div", { class: "task-card__name", text: t.name }),
        el("p", { class: "task-card__summary", text: t.summary })
      ])
    ]);
    card.addEventListener("click", () => openTask(t.id));
    return card;
  }

  function quadHead(move, couple) {
    return el("div", { class: "quad__head" }, [
      el("span", null, [el("span", { class: "axis-word axis-word--move", text: "Base movement" }), " ", `${LEVEL[move]}${ARROW[move]}`]),
      el("span", null, [el("span", { class: "axis-word axis-word--couple", text: "Physical coupling" }), " ", `${LEVEL[couple]}${ARROW[couple]}`])
    ]);
  }

  function renderTaxonomy(root) {
    // Reading order matches Figure 2: coupling increases upward, movement to the right.
    const quads = [["low", "high"], ["high", "high"], ["low", "low"], ["high", "low"]];
    const grid = el("div", { class: "taxo__grid" }, [
      el("span", { class: "taxo__axis taxo__axis--y", "aria-hidden": "true" }),
      el("span", { class: "taxo__axis taxo__axis--x", "aria-hidden": "true" }),
      el("span", { class: "taxo__label taxo__label--y", "aria-hidden": "true", text: "Physical coupling" }),
      el("span", { class: "taxo__label taxo__label--x", "aria-hidden": "true", text: "Base movement" })
    ]);
    for (const [move, couple] of quads) {
      const tasks = D.tasks.filter((t) => t.movement === move && t.coupling === couple);
      grid.append(el("div", { class: "quad", role: "group", "aria-label": `${LEVEL[move]} base movement, ${LEVEL[couple]} physical coupling` }, [
        quadHead(move, couple),
        el("div", { class: "quad__cards" }, tasks.map(taskCard))
      ]));
    }
    root.append(grid);
    $("[data-trio]").append(...D.tasks.filter((t) => t.robots === 3).map(taskCard));
  }

  /* ---------- Phase boxes (evaluation section and dialog) ---------- */
  function phaseBox(p, i, tag = "div") {
    const kind = kindOf(p);
    return el(tag, { class: "phase phase--" + kind, role: tag === "div" ? "listitem" : null }, [
      el("div", { class: "phase__top" }, [el("span", { class: "phase__p", text: `p${i + 1}` }), el("span", { class: "phase__name", text: p.name })]),
      kind === "n" ? null : el("span", { class: "phase__kind", text: kind === "c" ? "Collaborative" : "Non-collaborative" }),
      el("p", { class: "phase__cond", text: p.cond })
    ]);
  }

  function phaseFlow(t) {
    const flow = el("div", { class: "phase-flow", role: "list", "aria-label": `Phases of ${t.name}` });
    t.phases.forEach((p, i) => {
      if (i) flow.append(el("div", { class: "phase-flow__arrow", "aria-hidden": "true" }, icon("arrow")));
      flow.append(phaseBox(p, i));
    });
    return t.phaseNote ? el("div", null, [flow, phaseNote(t)]) : flow;
  }

  const phaseNote = (t) => el("p", { class: "phase-note", text: t.phaseNote });

  function renderPhaseExample(root) {
    const title = el("h3");
    const tabs = el("div", { class: "seg seg--left", role: "tablist", "aria-label": "Task" });
    const holder = el("div");
    const show = (id) => {
      const t = taskById[id];
      title.textContent = `Phases of ${t.name}`;
      tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.task === id)));
      holder.replaceChildren(phaseFlow(t));
    };
    D.tasks.filter((t) => t.robots === 2).forEach((t) => {
      const b = el("button", { type: "button", role: "tab", "data-task": t.id, text: t.name });
      b.addEventListener("click", () => show(t.id));
      tabs.append(b);
    });
    root.append(el("div", { class: "phase-example__head" }, [title, tabs]), holder);
    show("handover");
  }

  /* ---------- Task dialog ---------- */
  const dialog = $("[data-task-dialog]");
  let current = null;
  let returnFocus = null;

  function instructionTabs(t) {
    const text = el("p", { class: "instr-text", "aria-live": "polite" });
    const tabs = el("div", { class: "seg seg--left", role: "tablist", "aria-label": "Instruction role" });
    const select = (role) => {
      tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.role === role)));
      text.textContent = t.instructions[role];
    };
    for (const role of Object.keys(t.instructions)) {
      const b = el("button", { type: "button", role: "tab", "data-role": role, text: role });
      b.addEventListener("click", () => select(role));
      tabs.append(b);
    }
    select("Joint");
    return el("div", null, [tabs, text]);
  }

  function buildDialog(t) {
    const idx = D.tasks.findIndex((x) => x.id === t.id);
    const prev = D.tasks[(idx - 1 + D.tasks.length) % D.tasks.length];
    const next = D.tasks[(idx + 1) % D.tasks.length];

    const video = el("video", { src: taskVideoSrc(t.id, false), poster: posterOf(t.id), muted: true, loop: true, playsinline: true,
      autoplay: !H.reduceMotion.matches, controls: true, preload: "auto", "aria-label": `${t.name} demonstration video` });
    video.muted = true;

    const tags = [el("span", { class: "chip chip--solid", text: `${t.robots} humanoids` })];
    if (t.robots === 2) tags.push(quadHead(t.movement, t.coupling));
    const close = el("button", { type: "button", class: "icon-btn", "aria-label": "Close" }, icon("close"));
    close.addEventListener("click", () => dialog.close());

    const nav = (task, dir) => {
      const b = el("button", { type: "button", class: "link-btn" }, dir < 0 ? [icon("left"), task.name] : [task.name, icon("right")]);
      b.addEventListener("click", () => openTask(task.id));
      return b;
    };

    dialog.replaceChildren(
      el("header", { class: "td-head" }, [el("h2", { id: "task-dialog-title", text: t.name }), el("div", { class: "td-tags" }, tags), close]),
      el("div", { class: "td-scroll" }, [
        el("div", { class: "td-grid" }, [
          el("div", null, [
            el("div", { class: "td-video" }, video),
            el("p", { class: "td-summary", text: t.summary }),
            el("p", { class: "td-meta", text: `${t.scene} · ${t.objects}` })
          ]),
          el("div", null, [
            el("h3", { class: "td-section-title", text: "Scored phases" }),
            el("ol", { class: "td-phases" }, t.phases.map((p, i) => phaseBox(p, i, "li"))),
            t.phaseNote ? phaseNote(t) : null,
            el("h3", { class: "td-section-title", text: "Language instructions" }),
            instructionTabs(t)
          ])
        ]),
        el("div", { class: "td-results" }, [
          el("h3", { class: "td-section-title", text: `Phase-wise results on ${t.name} (%)` }),
          H.phaseChart(t.id, true)
        ])
      ]),
      el("footer", { class: "td-foot" }, [nav(prev, -1), nav(next, 1)])
    );
  }

  function openTask(id) {
    const t = taskById[id];
    if (!t) return;
    current = id;
    buildDialog(t);
    if (!dialog.open) {
      returnFocus = document.activeElement;
      dialog.showModal();
      document.documentElement.style.overflow = "hidden";
    }
    $(".td-scroll", dialog).scrollTop = 0;
    history.replaceState(null, "", "#task-" + id);
  }

  dialog.addEventListener("close", () => {
    const v = $("video", dialog);
    if (v) v.pause();
    H.tooltip.hide();
    document.documentElement.style.overflow = "";
    history.replaceState(null, "", location.pathname + location.search);
    current = null;
    if (returnFocus && returnFocus.focus) returnFocus.focus({ preventScroll: true });
  });
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  dialog.addEventListener("keydown", (e) => {
    if (!current || e.target.closest("[role=tablist]") || e.target.tagName === "VIDEO") return;
    const idx = D.tasks.findIndex((x) => x.id === current);
    if (e.key === "ArrowRight") openTask(D.tasks[(idx + 1) % D.tasks.length].id);
    if (e.key === "ArrowLeft") openTask(D.tasks[(idx - 1 + D.tasks.length) % D.tasks.length].id);
  });

  /* ---------- Egocentric viewer ---------- */
  function renderEgo(root) {
    const stage = el("div", { class: "ego-stage" });
    const caption = el("p", { class: "ego-caption" });
    let video = null;
    let first = true;

    const tabs = H.taskTabs((id) => show(id), { center: true });
    function show(id) {
      const t = taskById[id];
      const userAction = !first;
      first = false;
      const views = ["Robot A · ego", "Robot B · ego"];
      if (t.robots === 3) views.push("Robot C · ego");
      tabs.select(id);
      if (video) videos.unobserve(video);
      video = makeVideo({ src: `assets/video/ego/${id}.mp4`, poster: `assets/img/posters/ego-${id}.webp`, lazy: !userAction,
        label: `${t.name}: synchronized egocentric camera views of each humanoid` });
      // The strip puts the 4:3 ego cameras side by side, 854×640 each.
      stage.style.setProperty("--ego-ratio", `${854 * views.length} / 640`);
      stage.style.setProperty("--ego-cols", `repeat(${views.length}, minmax(0, 1fr))`);
      stage.style.setProperty("--ego-min", `${Math.round((854 * views.length) / 640 * 200)}px`);
      stage.replaceChildren(video, el("div", { class: "ego-labels", "aria-hidden": "true" },
        views.map((v) => el("span", { text: v }))));
      if (userAction) videos.play(video);
      caption.textContent = `${t.name} · human teleoperation demonstration, rendered at high resolution · policies observe 320×240 per camera`;
    }
    root.append(tabs.root, el("div", { class: "ego-scroll" }, stage), el("p", { class: "ego-hint", text: "Swipe sideways to see every camera →" }), caption);
    show("handover");
  }

  /* ---------- Failure carousel (Figure 5) ---------- */
  // Endless: the ten cards sit between two copies of themselves, and whenever the
  // scroll settles inside a copy it jumps by one set width to the same card in the middle.
  function renderFailures(root) {
    const track = el("div", { class: "carousel__track", tabindex: "0", "aria-label": "Failure cases, scroll horizontally" });
    const card = (t, copy) => {
      const name = el("button", { type: "button", class: "link-btn", style: "padding:0;font:650 17px/1.2 var(--font-mono);color:var(--ink)" }, [t.name, icon("arrow")]);
      name.addEventListener("click", () => openTask(t.id));
      if (copy) name.tabIndex = -1;
      const media = (D.failureVideos || []).includes(t.id)
        ? el("div", { class: "fail-card__video" }, makeVideo({
          src: `assets/video/failures/${t.id}.mp4`, poster: `assets/img/failures/${t.id}-poster.webp`,
          label: copy ? null : `${t.name} failure rollout video`
        }))
        : el("div", { class: "fail-card__frames" }, ["a", "b"].map((side, i) => el("figure", null, [
          el("img", { src: `assets/img/failures/${t.id}-${side}.webp`, alt: copy ? "" : `${t.name} failure, ${i === 0 ? "earlier" : "later"} moment`, loading: "lazy", width: "800", height: "600" }),
          el("figcaption", { text: i === 0 ? "Earlier" : "Later" })
        ])));
      return el("article", { class: "fail-card", "aria-hidden": copy ? "true" : null }, [
        media,
        el("div", { class: "fail-card__body" }, [el("div", { class: "fail-card__name" }, name), el("p", { class: "fail-card__text", text: D.failures[t.id] })])
      ]);
    };
    const n = D.tasks.length;
    const cards = [true, false, true].flatMap((copy) => D.tasks.map((t) => card(t, copy)));
    track.append(...cards);
    const count = el("span", { class: "carousel__count" });
    const prev = el("button", { type: "button", class: "icon-btn", "aria-label": "Previous" }, icon("left"));
    const next = el("button", { type: "button", class: "icon-btn", "aria-label": "Next" }, icon("right"));
    const step = () => cards[1].offsetLeft - cards[0].offsetLeft;
    // Scroll position that puts the first middle card at the left edge.
    const start = () => cards[n].offsetLeft - cards[0].offsetLeft + (parseFloat(getComputedStyle(track).paddingLeft) || 0);
    const index = () => Math.round((track.scrollLeft - start()) / step());
    const wrap = (i) => ((i % n) + n) % n;
    const jumpTo = (i) => {
      track.style.scrollBehavior = "auto";
      track.scrollLeft = start() + i * step();
      track.style.scrollBehavior = "";
    };
    let current = 0, target = null;
    const update = () => { current = wrap(index()); count.textContent = `${current + 1} / ${n}`; };
    // Back into the middle copy once the scroll has settled (a jump mid-animation would stop it).
    const recenter = () => {
      target = null;
      const i = index();
      if (i < 0 || i >= n) jumpTo(wrap(i));
      update();
    };
    const behavior = () => (H.reduceMotion.matches ? "auto" : "smooth");
    // Quick repeated clicks add up: each one moves the pending target, not the mid-animation position.
    const go = (d) => {
      const i = (target ?? index()) + d;
      if (i < -n || i >= 2 * n) return;
      target = i;
      track.scrollTo({ left: start() + i * step(), behavior: behavior() });
    };
    prev.addEventListener("click", () => go(-1));
    next.addEventListener("click", () => go(1));
    let raf = 0, settle = 0;
    const hasScrollEnd = "onscrollend" in window;
    track.addEventListener("scroll", () => {
      cancelAnimationFrame(raf); raf = requestAnimationFrame(update);
      if (!hasScrollEnd) { clearTimeout(settle); settle = setTimeout(recenter, 160); }
    }, { passive: true });
    if (hasScrollEnd) track.addEventListener("scrollend", recenter);
    // Card widths change with the viewport: keep the same card at the left edge.
    let resizeRaf = 0;
    window.addEventListener("resize", () => {
      cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(() => { jumpTo(current); update(); });
    });
    root.append(el("div", { class: "carousel" }, [track, el("div", { class: "carousel__nav" }, [count, prev, next])]));
    requestAnimationFrame(() => { jumpTo(0); update(); });
  }

  H.openTask = openTask;

  document.addEventListener("DOMContentLoaded", () => {
    renderTaxonomy($("[data-taxonomy]"));
    renderEgo($("[data-ego]"));
    renderPhaseExample($("[data-phase-example]"));
    renderFailures($("[data-failures]"));
    // Deep links: #task-<id> opens that task, on load and on in-page hash changes.
    const openFromHash = () => {
      const m = location.hash.match(/^#task-([a-z]+)$/);
      if (m && taskById[m[1]]) openTask(m[1]);
    };
    openFromHash();
    window.addEventListener("hashchange", openFromHash);
  });
})();
