const header = document.querySelector("[data-header]");

if (header) {
  const updateHeader = () => header.classList.toggle("is-scrolled", window.scrollY > 90);
  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });
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
      selection.removeAllRanges();
      selection.addRange(range);
      if (label) label.textContent = "Select & copy";
    }
  });
}
