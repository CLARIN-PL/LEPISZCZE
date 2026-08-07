(() => {
  const featuredRoot = document.querySelector("[data-pipeline-featured]");
  const catalogueRoot = document.querySelector("[data-pipeline-catalogue]");
  const status = document.querySelector("[data-pipeline-status]");
  const filters = document.querySelector("[data-pipeline-filters]");

  if (!featuredRoot || !catalogueRoot || !status || !filters) return;

  const maturityLabels = {
    benchmark_candidate: "Benchmark candidate",
    public_preview: "Public preview",
    public_dataset: "Public dataset",
  };

  const typeLabels = {
    benchmark_dataset: "Benchmark dataset",
    enriched_corpus: "Enriched corpus",
    graph_dataset: "Graph dataset",
    instruction_dataset: "Instruction dataset",
    source_corpus: "Source corpus",
  };

  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const externalLink = (label, href) => {
    const link = element("a", "pipeline-link");
    link.href = href;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.append(label, " ", element("span", "", "↗"));
    return link;
  };

  const renderFeaturedCard = (dataset) => {
    const card = element("article", `pipeline-card pipeline-${dataset.project.toLowerCase()}`);
    const top = element("div", "pipeline-card-topline");
    const statusLabel = element("span", `pipeline-status-label maturity-${dataset.maturity}`);
    statusLabel.append(element("i"), maturityLabels[dataset.maturity]);
    top.append(statusLabel, element("span", "", dataset.project));

    const meta = element("p", "pipeline-card-meta", `${typeLabels[dataset.artifact_type]} · ${dataset.language}`);
    const title = element("h4", "", dataset.name);
    const summary = element("p", "pipeline-card-summary", dataset.summary);
    const size = element("p", "pipeline-card-size", dataset.size_label);

    const gate = element("div", "pipeline-gate");
    gate.append(
      element("strong", "", dataset.validation.missing.length ? "Before leaderboard" : "Leaderboard ready"),
      element("span", "", dataset.validation.missing.join(" · ") || "All promotion gates passed"),
    );

    const links = element("div", "pipeline-card-links");
    links.append(externalLink("View dataset", dataset.links.dataset));
    if (dataset.links.code) links.append(externalLink("Code", dataset.links.code));
    if (dataset.links.paper) links.append(externalLink("Paper", dataset.links.paper));

    card.append(top, meta, title, summary, size, gate, links);
    return card;
  };

  const renderCatalogue = (datasets, project) => {
    const visible = project === "all" ? datasets : datasets.filter((dataset) => dataset.project === project);
    const projects = [...new Set(visible.map((dataset) => dataset.project))];

    catalogueRoot.replaceChildren(
      ...projects.map((projectName) => {
        const group = element("section", "catalogue-group");
        const projectDatasets = visible.filter((dataset) => dataset.project === projectName);
        const heading = element("div", "catalogue-group-heading");
        heading.append(
          element("h4", "", projectName),
          element("span", "", `${projectDatasets.length} datasets`),
        );

        const list = element("div", "catalogue-list");
        projectDatasets.forEach((dataset) => {
          const row = externalLink(dataset.name, dataset.links.dataset);
          row.className = "catalogue-row";
          row.setAttribute("aria-label", `${dataset.name} on Hugging Face`);

          const identity = element("span", "catalogue-identity");
          identity.append(
            element("strong", "", dataset.name),
            element("small", "", dataset.scope),
          );
          const tags = element("span", "catalogue-tags");
          tags.append(
            element("span", "", typeLabels[dataset.artifact_type]),
            element("span", `maturity-text maturity-${dataset.maturity}`, maturityLabels[dataset.maturity]),
          );
          row.replaceChildren(identity, tags, element("span", "catalogue-arrow", "↗"));
          list.append(row);
        });

        group.append(heading, list);
        return group;
      }),
    );
  };

  const initialise = async () => {
    try {
      const response = await fetch("data/datasets.json");
      if (!response.ok) throw new Error(`Dataset registry returned ${response.status}`);
      const registry = await response.json();
      const datasets = registry.datasets;

      document.querySelector("[data-pipeline-stat='datasets']").textContent = datasets.length;
      document.querySelector("[data-pipeline-stat='candidates']").textContent = datasets.filter(
        (dataset) => dataset.maturity === "benchmark_candidate",
      ).length;

      featuredRoot.replaceChildren(
        ...datasets.filter((dataset) => dataset.featured).map(renderFeaturedCard),
      );
      renderCatalogue(datasets, "all");

      status.hidden = true;
      featuredRoot.hidden = false;
      catalogueRoot.hidden = false;

      filters.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-filter]");
        if (!button) return;
        filters.querySelectorAll("button").forEach((item) => {
          item.setAttribute("aria-pressed", String(item === button));
        });
        renderCatalogue(datasets, button.dataset.filter);
      });
    } catch (error) {
      console.error("Unable to load dataset registry", error);
      status.classList.add("is-error");
      status.textContent = "The dataset registry could not be loaded. Browse AITAX and JuDDGES directly on Hugging Face.";
    }
  };

  initialise();
})();
