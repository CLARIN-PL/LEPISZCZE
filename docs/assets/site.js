const header = document.querySelector("[data-header]");

if (header) {
  const updateHeader = () => header.classList.toggle("is-scrolled", window.scrollY > 90);
  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });
}

const navToggle = document.querySelector("[data-nav-toggle]");

if (header && navToggle) {
  const closeNavigation = () => {
    header.classList.remove("is-menu-open");
    navToggle.setAttribute("aria-expanded", "false");
    navToggle.textContent = "Menu";
  };

  navToggle.addEventListener("click", () => {
    const open = !header.classList.contains("is-menu-open");
    header.classList.toggle("is-menu-open", open);
    navToggle.setAttribute("aria-expanded", String(open));
    navToggle.textContent = open ? "Close" : "Menu";
  });

  header.querySelectorAll(".site-nav a").forEach((link) => {
    link.addEventListener("click", closeNavigation);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && header.classList.contains("is-menu-open")) {
      closeNavigation();
      navToggle.focus();
    }
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 960) closeNavigation();
  });
}

const copyButton = document.querySelector("[data-copy-button]");
const citation = document.querySelector("#bibtex");

if (copyButton && citation) {
  copyButton.addEventListener("click", async () => {
    const label = copyButton.querySelector("[data-copy-label]");

    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      if (label) label.textContent = "Copied";
      window.setTimeout(() => {
        if (label) label.textContent = "Copy BibTeX";
      }, 1800);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      if (selection) {
        selection.removeAllRanges();
        selection.addRange(range);
      }
      if (label) label.textContent = "Select & copy";
    }
  });
}
