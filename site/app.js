document.addEventListener("DOMContentLoaded", () => {
  // 수식
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "table"],
      throwOnError: false,
    });
  }

  // 탭: 바뀐 부분 / 파일 전체
  document.querySelectorAll(".tabs").forEach((tabs) => {
    const block = tabs.closest(".block");
    tabs.querySelectorAll("button[data-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        tabs.querySelectorAll("button[data-tab]").forEach((b) => b.setAttribute("aria-selected", b === btn));
        block.querySelectorAll(".tabpanel").forEach((p) => (p.hidden = p.dataset.panel !== btn.dataset.tab));
      });
    });
  });

  // 접힌 줄 펼치기
  document.querySelectorAll(".diff tr.fold button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const fold = btn.closest("tr");
      const hidden = fold.nextElementSibling;
      if (hidden && hidden.classList.contains("hidden")) hidden.classList.remove("hidden");
      fold.remove();
    });
  });

  // 모바일에서는 목차를 접어 둔다
  const toc = document.querySelector(".toc");
  if (toc && window.matchMedia("(max-width: 900px)").matches) toc.open = false;

  // 차트: 화면에 들어올 때 불러온다
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const load = (el) => {
    if (el.dataset.loaded) return;
    el.dataset.loaded = "1";
    const tryEmbed = () => {
      if (!window.vegaEmbed) return setTimeout(tryEmbed, 100);
      const target = document.createElement("div");
      vegaEmbed(target, el.dataset.spec, { actions: false, theme: dark ? "dark" : undefined })
        .then(() => el.replaceChildren(target))
        .catch((e) => (el.textContent = "차트를 불러오지 못했다: " + e.message));
    };
    tryEmbed();
  };
  const charts = document.querySelectorAll(".chart[data-spec]");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          load(e.target);
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: "400px" });
    charts.forEach((c) => io.observe(c));
  } else charts.forEach(load);

  // ← → 로 단계 이동
  const main = document.querySelector("main");
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) return;
    if (e.key === "ArrowLeft" && main.dataset.prev) location.href = main.dataset.prev;
    if (e.key === "ArrowRight" && main.dataset.next) location.href = main.dataset.next;
  });
});
