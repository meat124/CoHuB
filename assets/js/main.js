/*
 * Shared helpers, the hero (opening shot + mosaic), navigation, video playback
 * and the config-driven parts of the page (authors, links, BibTeX).
 * results.js and tasks.js use window.CoHuB defined here.
 */
(function () {
  "use strict";

  const D = window.COHUB_DATA;
  const C = window.COHUB_CONFIG;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const isSmall = () => window.matchMedia("(max-width: 767px)").matches;
  const SVG_NS = "http://www.w3.org/2000/svg";

  /* ---------- DOM helpers ---------- */
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key === "style") node.style.cssText = value;
        else node.setAttribute(key, value === true ? "" : value);
      }
    }
    for (const child of [].concat(children || [])) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function icon(name, cls) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "icon" + (cls ? " " + cls : ""));
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", "#i-" + name);
    svg.append(use);
    return svg;
  }

  /* ---------- Data lookups ---------- */
  const taskById = Object.fromEntries(D.tasks.map((t) => [t.id, t]));
  const methodById = Object.fromEntries(D.methods.map((m) => [m.id, m]));
  const familyById = Object.fromEntries(D.families.map((f) => [f.id, f]));

  function methodLabel(m) {
    const frag = document.createDocumentFragment();
    frag.append(m.name);
    if (m.sub) frag.append(el("sub", { text: m.sub }));
    return frag;
  }
  function methodText(m) { return m.name + (m.sub || ""); }

  const LEVEL = { low: "Low", high: "High" };
  function taskMeta(t) {
    if (t.robots === 3) return "3 humanoids";
    return `${LEVEL[t.movement]} movement · ${LEVEL[t.coupling]} coupling`;
  }

  /* ---------- Tooltip (values are always visible elsewhere; this only adds context) ---------- */
  const tip = $("[data-tooltip]");
  const tooltip = {
    show(content, x, y) {
      tip.replaceChildren(content);
      tip.hidden = false;
      const pad = 14;
      const { width, height } = tip.getBoundingClientRect();
      let left = x + pad, top = y + pad;
      if (left + width > window.innerWidth - 8) left = x - width - pad;
      if (top + height > window.innerHeight - 8) top = y - height - pad;
      tip.style.left = Math.max(8, left) + "px";
      tip.style.top = Math.max(8, top) + "px";
    },
    hide() { tip.hidden = true; }
  };

  /* ---------- Video playback ----------
   * Lazy videos get their source when they approach the viewport. Autoplaying
   * videos play only while visible (and never with reduced motion). */
  function taskVideoSrc(taskId, small) {
    return `assets/video/${small ? "tasks-sm" : "tasks"}/${taskId}.mp4`;
  }
  function makeVideo({ src, poster, lazy = true, autoplay = true, label }) {
    const video = el("video", {
      muted: true, loop: true, playsinline: true, preload: lazy ? "none" : "auto",
      poster, "aria-label": label || null, "aria-hidden": label ? null : "true"
    });
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    if (lazy) video.dataset.src = src;
    else video.src = src;
    if (autoplay) video.dataset.autoplay = "";
    videos.observe(video);
    return video;
  }

  const videos = (function () {
    const loader = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const v = entry.target;
        if (v.dataset.src) {
          v.src = v.dataset.src;
          delete v.dataset.src;
          v.preload = "auto";
        }
        loader.unobserve(v);
      }
    }, { rootMargin: "400px 0px" });

    const inView = new Set();
    const player = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const v = entry.target;
        if (entry.isIntersecting) inView.add(v); else inView.delete(v);
        if (entry.isIntersecting && !reduceMotion.matches) play(v);
        else v.pause();
      }
    }, { threshold: 0.2 });
    // Browsers may pause muted videos in background tabs; resume the visible ones on return.
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && !reduceMotion.matches) inView.forEach((v) => { if (v.isConnected) play(v); });
    });

    function play(v) {
      if (v.dataset.src) { v.src = v.dataset.src; delete v.dataset.src; }
      const p = v.play();
      if (p && p.catch) p.catch(() => {});
    }

    return {
      observe(v) {
        if (v.dataset.src) loader.observe(v);
        if (v.dataset.autoplay !== undefined) player.observe(v);
      },
      unobserve(v) { loader.unobserve(v); player.unobserve(v); inView.delete(v); },
      play
    };
  })();

  /* ---------- Hero mosaic: rows of 4 / 2 / 4 ---------- */
  function buildHero() {
    const small = isSmall();
    const layout = D.heroLayout;
    const intro = D.heroIntro || {};
    const tiles = {};
    let order = 0;
    const makeTile = (id, right) => {
      const t = taskById[id];
      const src = !small && intro.src && intro.task === id ? intro.src : taskVideoSrc(id, small);
      const tile = el("button", {
        type: "button", class: "tile" + (right ? " tile--right" : ""), "data-task": id, style: `--i:${order++}`,
        "aria-label": `${t.name}: ${t.summary} Open task details.`
      }, [
        makeVideo({ src, poster: `assets/img/posters/${id}.webp`, lazy: false }),
        t.robots === 3 ? el("span", { class: "tile__badge", text: "3 humanoids" }) : null,
        el("span", { class: "tile__label" }, [
          el("span", { class: "tile__name", text: t.name }),
          el("span", { class: "tile__meta", text: taskMeta(t) })
        ])
      ]);
      tile.addEventListener("click", () => window.CoHuB.openTask && window.CoHuB.openTask(id));
      tiles[id] = tile;
      return tile;
    };
    layout.top.forEach((id) => $('[data-hero-row="top"]').append(makeTile(id)));
    layout.middle.forEach((id, i) => $('[data-hero-row="middle"]').append(makeTile(id, i === 1)));
    layout.bottom.forEach((id) => $('[data-hero-row="bottom"]').append(makeTile(id)));
    return tiles;
  }

  /* ---------- Opening shot: one task full screen, then the mosaic assembles ---------- */
  function setupIntro(tiles) {
    const root = document.documentElement;
    if (!root.classList.contains("hero-intro")) return;
    window.__cohubIntro = true;
    const cfg = D.heroIntro || {};
    const tile = tiles[cfg.task];
    if (!tile || window.scrollY > 40) { root.classList.remove("hero-intro", "hero-locked"); return; }

    const hero = $(".hero");
    const grid = $("[data-hero-grid]");
    const progress = $("[data-hero-progress]");
    const video = $("video", tile);
    const placeholder = el("div", { class: "tile tile--placeholder", "aria-hidden": "true" });
    tile.replaceWith(placeholder);
    tile.classList.add("tile--intro");
    grid.append(tile);
    videos.play(video);

    const hold = cfg.holdMs || 4200;
    let active = true;
    let timer = 0;
    const startTimer = () => {
      if (timer || !active) return;
      progress.style.transition = `transform ${hold}ms linear`;
      requestAnimationFrame(() => requestAnimationFrame(() => { progress.style.transform = "scaleX(1)"; }));
      timer = setTimeout(assemble, hold);
    };
    video.addEventListener("playing", startTimer, { once: true });
    setTimeout(startTimer, 1500); // do not wait on a slow network

    function assemble() {
      if (!active) return;
      active = false;
      clearTimeout(timer);
      const g = grid.getBoundingClientRect();
      const r = placeholder.getBoundingClientRect();
      const box = (b) => ({ top: b.top + "px", left: b.left + "px", width: b.width + "px", height: b.height + "px" });
      const from = box({ top: 0, left: 0, width: g.width, height: g.height });
      const to = box({ top: r.top - g.top, left: r.left - g.left, width: r.width, height: r.height });
      // 1) the other nine tiles land while this one shrinks into its cell,
      // 2) then the title card, tile labels and scroll cue appear,
      // 3) then hover details are switched on.
      root.classList.add("hero-assembling");
      root.classList.remove("hero-intro");
      progress.style.transition = "opacity 0.4s ease";
      const later = (ms, fn) => (document.hidden ? fn() : setTimeout(fn, ms));
      later(1250, () => root.classList.remove("hero-assembling"));
      later(2150, () => root.classList.remove("hero-locked"));
      const finish = () => {
        placeholder.replaceWith(tile);
        tile.classList.remove("tile--intro");
        tile.style.top = tile.style.left = tile.style.width = tile.style.height = "";
        videos.play(video);
      };
      if (document.hidden || !tile.animate) { finish(); return; }
      Object.assign(tile.style, to);
      tile.animate([from, to], { duration: 1150, easing: "cubic-bezier(0.65, 0, 0.35, 1)" }).onfinish = finish;
    }

    // Any interaction skips straight to the mosaic.
    const skip = () => assemble();
    window.addEventListener("wheel", skip, { passive: true, once: true });
    window.addEventListener("touchmove", skip, { passive: true, once: true });
    window.addEventListener("keydown", skip, { once: true });
    window.addEventListener("scroll", skip, { passive: true, once: true });
    hero.addEventListener("click", (e) => {
      if (!active) return;
      e.preventDefault();
      e.stopPropagation();
      assemble();
    }, true);
  }

  /* ---------- Top navigation: appears after the hero, highlights the current section ---------- */
  function setupNav() {
    const nav = $("[data-topnav]");
    new IntersectionObserver(([entry]) => {
      nav.classList.toggle("is-visible", !entry.isIntersecting);
    }, { threshold: 0.08 }).observe($(".hero"));

    const links = $$(".topnav__links a");
    const byId = Object.fromEntries(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const spy = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        links.forEach((a) => a.classList.remove("is-active"));
        const active = byId[entry.target.id];
        if (active) {
          active.classList.add("is-active");
          active.scrollIntoView({ block: "nearest", inline: "nearest" });
        }
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(byId).forEach((id) => { const s = document.getElementById(id); if (s) spy.observe(s); });

    const bar = $("[data-progress]");
    let ticking = false;
    window.addEventListener("scroll", () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        ticking = false;
        const max = document.documentElement.scrollHeight - window.innerHeight;
        bar.style.transform = `scaleX(${max > 0 ? Math.min(1, window.scrollY / max) : 0})`;
      });
    }, { passive: true });
  }

  /* ---------- Config: venue, authors, links, BibTeX ---------- */
  function linkButton(link, available) {
    if (available) {
      return el("a", { class: "btn", href: link.url, target: "_blank", rel: "noopener" }, [icon(link.icon), link.label]);
    }
    return el("span", { class: "btn btn--soon", "aria-disabled": "true", title: "Coming soon" },
      [icon(link.icon), link.label, el("span", { class: "btn__soon", text: "soon" })]);
  }

  function renderConfig() {
    $$("[data-venue]").forEach((n) => {
      n.textContent = C.venue || "";
      n.hidden = !C.venue;
    });

    const authorsBox = $("[data-authors]");
    if (C.authors && C.authors.length) {
      const notes = C.authorNotes || {};
      const lines = C.authors.every(Array.isArray) ? C.authors : [C.authors];
      const everyone = lines.flat();
      const author = (a) => {
        const marks = (a.affiliations || []).join(",") + (a.notes || []).map((k) => notes[k].mark).join("");
        const name = a.url ? el("a", { href: a.url, target: "_blank", rel: "noopener", text: a.name }) : el("span", { text: a.name });
        const equal = (a.notes || []).includes("equal");
        return el("span", { class: equal ? "author author--equal" : "author" }, [name, marks ? el("sup", { text: marks }) : null]);
      };
      const list = el("div", { class: "authors__list" }, lines.map((line) => el("div", { class: "authors__line" }, line.map(author))));
      const affils = el("div", { class: "authors__affil" },
        (C.affiliations || []).map((name, i) => el("span", null, [el("sup", { text: String(i + 1) }), " ", name])));
      const used = Object.keys(notes).filter((k) => everyone.some((a) => (a.notes || []).includes(k)));
      const legend = used.length
        ? el("p", { class: "authors__note" }, used.map((k) => el("span", null, [el("sup", { text: notes[k].mark }), " ", notes[k].text])))
        : null;
      authorsBox.replaceChildren(...[list, affils, legend].filter(Boolean));
    }

    const titleRow = $('[data-links="title"]');
    const dataRow = $('[data-links="data"]');
    const heroRow = $("[data-hero-links]");
    for (const link of C.links) {
      const ok = Boolean(link.url);
      const row = link.where === "data" ? dataRow : titleRow;
      if (ok || C.showComingSoon) row.append(linkButton(link, ok));
      if (ok && link.where !== "data") heroRow.append(linkButton(link, true));
    }
    // An empty BibTeX (before arXiv) shows "soon" instead of an entry.
    const bib = C.bibtex;
    if (bib) titleRow.append(el("a", { class: "btn", href: "#citation" }, [icon("quote"), "BibTeX"]));
    else if (C.showComingSoon) titleRow.append(linkButton({ icon: "quote", label: "BibTeX" }, false));

    if (!bib) {
      $(".bibtex").replaceChildren(el("p", { class: "bibtex__soon" },
        [el("span", { class: "btn__soon", text: "soon" }), "The BibTeX entry will be posted here once the paper is on arXiv."]));
    } else {
      $("[data-bibtex]").textContent = bib;
      const copy = $("[data-copy-bibtex]");
      copy.addEventListener("click", async () => {
        const label = $("span", copy);
        try {
          await navigator.clipboard.writeText(bib);
          label.textContent = "Copied";
        } catch (err) {
          const range = document.createRange();
          range.selectNodeContents($("[data-bibtex]"));
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          label.textContent = "Selected";
        }
        setTimeout(() => { label.textContent = "Copy"; }, 1600);
      });
    }

    const ack = $("[data-acknowledgements]");
    if (C.acknowledgements) {
      ack.textContent = C.acknowledgements;
      ack.hidden = false;
    }
  }

  /* ---------- Reveal on scroll, count-up numbers ---------- */
  function onceVisible(nodes, callback, options) {
    const io = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        io.unobserve(entry.target);
        callback(entry.target);
      }
    }, options);
    nodes.forEach((n) => io.observe(n));
  }

  function setupReveal() {
    const items = $$(".reveal");
    if (reduceMotion.matches || !("IntersectionObserver" in window)) {
      items.forEach((n) => n.classList.add("is-in"));
      return;
    }
    onceVisible(items, (n) => n.classList.add("is-in"), { threshold: 0.06, rootMargin: "0px 0px -40px 0px" });
  }

  function setupCountUp() {
    const nodes = $$("[data-count]");
    if (reduceMotion.matches || !nodes.length) return;
    nodes.forEach((n) => { n.textContent = "0"; });
    onceVisible(nodes, (n) => {
      const target = Number(n.dataset.count);
      const t0 = performance.now();
      const step = (t) => {
        const k = Math.min(1, (t - t0) / 1300);
        n.textContent = String(Math.round(target * (1 - Math.pow(1 - k, 3))));
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }, { threshold: 0.6 });
  }

  /* ---------- Figure 3: hovering a step lifts it and blurs the rest ----------
   * Each step gets its own copy of the figure, clipped to that step's box. The
   * active copy rises with a soft shadow over the blurred base image, so moving
   * between steps crossfades instead of sliding. Hover, focus or tap a step
   * (or its label below). */
  function setupPipeline() {
    const fig = $("[data-pipeline]");
    if (!fig) return;
    const base = $(".pipeline__img", fig);
    const hots = $$(".pipeline__hot", fig);
    const labels = $$(".pipeline__steps [data-step]", fig);
    const pct = (node, name) => parseFloat(node.style.getPropertyValue(name)) || 0;
    const layers = hots.map((h) => {
      const l = pct(h, "--l"), r = pct(h, "--r");
      const img = el("img", { src: base.getAttribute("src"), srcset: base.getAttribute("srcset"), sizes: base.getAttribute("sizes"),
        width: base.getAttribute("width"), height: base.getAttribute("height"), alt: "", "aria-hidden": "true", decoding: "async" });
      img.style.clipPath = `inset(0.6% ${r}% 0.8% ${l}% round 12px)`;
      const layer = el("div", { class: "pipeline__layer", "data-step": h.dataset.step }, img);
      layer.style.transformOrigin = `${l + (100 - l - r) / 2}% 50%`;
      hots[0].before(layer);
      return layer;
    });
    let pinned = null;
    const show = (step) => {
      fig.classList.toggle("is-focus", hots.some((h) => h.dataset.step === step));
      [...layers, ...hots, ...labels].forEach((n) => n.classList.toggle("is-active", n.dataset.step === step));
    };
    for (const n of [...hots, ...labels]) {
      n.addEventListener("pointerenter", (e) => { if (e.pointerType !== "touch") show(n.dataset.step); });
      n.addEventListener("pointerleave", (e) => { if (e.pointerType !== "touch") show(pinned); });
      n.addEventListener("click", () => { pinned = pinned === n.dataset.step ? null : n.dataset.step; show(pinned); });
    }
    hots.forEach((h) => {
      h.addEventListener("focus", () => show(h.dataset.step));
      h.addEventListener("blur", () => show(pinned));
    });
  }

  /* ---------- Task tabs grouped by team size ---------- */
  // `extra` adds a first row of non-task choices, e.g. { label: "Average", items: [{ id, text }] }.
  function taskTabs(onSelect, { center = false, extra = null } = {}) {
    const root = el("div", { class: "seg-groups" + (center ? " seg-groups--center" : ""), role: "tablist", "aria-label": "Task" });
    const buttons = [];
    const group = (label, items) => {
      const list = el("div", { class: "seg" + (center ? "" : " seg--left") });
      for (const item of items) {
        const b = el("button", { type: "button", role: "tab", "data-task": item.id, text: item.text });
        b.addEventListener("click", () => onSelect(item.id));
        list.append(b);
        buttons.push(b);
      }
      root.append(el("div", { class: "seg-group" }, [el("span", { class: "seg-group__label", text: label }), list]));
    };
    if (extra) group(extra.label, extra.items);
    for (const n of [2, 3]) {
      group(`${n} humanoids`, D.tasks.filter((task) => task.robots === n).map((t) => ({ id: t.id, text: t.name })));
    }
    return {
      root,
      select(id) { buttons.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.task === id))); }
    };
  }

  window.CoHuB = {
    $, $$, el, icon, methodLabel, methodText, taskById, methodById, familyById, taskMeta,
    tooltip, makeVideo, taskVideoSrc, videos, isSmall, reduceMotion, onceVisible, taskTabs, openTask: null
  };

  document.addEventListener("DOMContentLoaded", () => {
    const tiles = buildHero();
    setupIntro(tiles);
    setupNav();
    renderConfig();
    setupReveal();
    setupCountUp();
    setupPipeline();
  });
})();
