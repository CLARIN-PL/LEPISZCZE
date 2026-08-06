(() => {
  "use strict";

  const panel = document.querySelector("[data-leaderboard-panel]");
  if (!panel) return;

  const elements = {
    panel,
    status: panel.querySelector("[data-status]"),
    content: panel.querySelector("[data-content]"),
    task: panel.querySelector("[data-task]"),
    dataset: panel.querySelector("[data-dataset]"),
    metric: panel.querySelector("[data-metric]"),
    search: panel.querySelector("[data-search]"),
    taskLabel: panel.querySelector("[data-task-label]"),
    datasetLabel: panel.querySelector("[data-dataset-label]"),
    resultCount: panel.querySelector("[data-result-count]"),
    tableCaption: panel.querySelector("[data-table-caption]"),
    tableHead: panel.querySelector("[data-table-head]"),
    tableBody: panel.querySelector("[data-table-body]"),
    empty: panel.querySelector("[data-empty]"),
    share: panel.querySelector("[data-share]"),
    sourceLink: panel.querySelector("[data-source-link]"),
  };

  const labels = {
    accuracy: "Accuracy",
    exact: "Exact match",
    f1: "F1",
    f1_macro: "F1 macro",
    f1_micro: "F1 micro",
    f1_weighted: "F1 weighted",
    precision_macro: "Precision macro",
    precision_micro: "Precision micro",
    precision_weighted: "Precision weighted",
    recall_macro: "Recall macro",
    recall_micro: "Recall micro",
    recall_weighted: "Recall weighted",
    HasAns_exact: "Has answer · exact",
    HasAns_f1: "Has answer · F1",
    NoAns_f1: "No answer · F1",
  };

  const defaultMetric = (metrics) => {
    for (const candidate of ["f1_macro", "f1", "accuracy", "f1_micro"]) {
      if (metrics.includes(candidate)) return candidate;
    }
    return metrics[0];
  };

  const state = {
    data: null,
    task: "",
    dataset: "",
    metric: "",
    search: "",
    direction: "desc",
  };

  const metricLabel = (metric) => labels[metric] || metric.replaceAll("_", " ");

  const modelParts = (model) => {
    const separator = model.indexOf("/");
    if (separator < 0) return { owner: "Baseline", name: model };
    return { owner: model.slice(0, separator), name: model.slice(separator + 1) };
  };

  const formatNumber = (value) => {
    if (!Number.isFinite(value)) return "—";
    return new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 3,
      maximumFractionDigits: 3,
    }).format(value);
  };

  const tasks = () => [...new Set(state.data.entries.map((entry) => entry.task))].sort();

  const datasetsForTask = () => {
    const entries = state.data.entries.filter((entry) => entry.task === state.task);
    const datasets = new Map();
    entries.forEach((entry) => {
      if (!datasets.has(entry.dataset)) {
        datasets.set(entry.dataset, {
          value: entry.dataset,
          label: entry.dataset_label,
          task: entry.task,
        });
      }
    });
    return [...datasets.values()].sort((a, b) => a.label.localeCompare(b.label));
  };

  const entriesForDataset = () =>
    state.data.entries.filter(
      (entry) => entry.task === state.task && entry.dataset === state.dataset,
    );

  const metricsForDataset = () => {
    const names = new Set();
    entriesForDataset().forEach((entry) => {
      Object.keys(entry.metrics).forEach((metric) => names.add(metric));
    });
    return [...names].sort((a, b) => metricLabel(a).localeCompare(metricLabel(b)));
  };

  const readUrlState = () => {
    const params = new URLSearchParams(window.location.search);
    state.task = params.get("task") || "";
    if (!state.task && params.get("track") === "expansio") state.task = "Question Answering";
    state.dataset = params.get("dataset") || "";
    state.metric = params.get("metric") || "";
    state.search = params.get("q") || "";
    state.direction = params.get("direction") === "asc" ? "asc" : "desc";
  };

  const writeUrlState = () => {
    const params = new URLSearchParams();
    if (state.task) params.set("task", state.task);
    if (state.dataset) params.set("dataset", state.dataset);
    if (state.metric) params.set("metric", state.metric);
    if (state.search) params.set("q", state.search);
    if (state.direction === "asc") params.set("direction", "asc");
    const query = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
  };

  const refreshTaskOptions = () => {
    const availableTasks = tasks();
    if (!availableTasks.includes(state.task)) state.task = availableTasks[0] || "";
    elements.task.replaceChildren(
      ...availableTasks.map((task) => new Option(task, task)),
    );
    elements.task.value = state.task;
  };

  const refreshDatasetOptions = () => {
    const datasets = datasetsForTask();
    if (!datasets.some((dataset) => dataset.value === state.dataset)) {
      state.dataset = datasets[0]?.value || "";
    }
    elements.dataset.replaceChildren(
      ...datasets.map((dataset) => new Option(dataset.label, dataset.value)),
    );
    elements.dataset.value = state.dataset;
  };

  const refreshMetricOptions = () => {
    const metrics = metricsForDataset();
    if (!metrics.includes(state.metric)) state.metric = defaultMetric(metrics);
    elements.metric.replaceChildren(
      ...metrics.map((metric) => new Option(metricLabel(metric), metric)),
    );
    elements.metric.value = state.metric;
  };

  const rankedEntries = () => {
    const query = state.search.trim().toLocaleLowerCase();
    const direction = state.direction === "desc" ? -1 : 1;
    return entriesForDataset()
      .filter((entry) => !query || entry.model.toLocaleLowerCase().includes(query))
      .sort((a, b) => {
        const aValue = a.metrics[state.metric]?.mean;
        const bValue = b.metrics[state.metric]?.mean;
        if (aValue === undefined && bValue === undefined) return a.model.localeCompare(b.model);
        if (aValue === undefined) return 1;
        if (bValue === undefined) return -1;
        if (aValue === bValue) return a.model.localeCompare(b.model);
        return (aValue - bValue) * direction;
      });
  };

  const renderTable = () => {
    const metrics = metricsForDataset();
    const entries = rankedEntries();
    const dataset = datasetsForTask().find((item) => item.value === state.dataset);

    elements.taskLabel.textContent = dataset?.task || "—";
    elements.datasetLabel.textContent = dataset?.label || "—";
    const totalEntries = entriesForDataset().length;
    elements.resultCount.textContent = state.search
      ? `${entries.length} of ${totalEntries} models`
      : `${entries.length} ${entries.length === 1 ? "model" : "models"}`;
    elements.tableCaption.textContent = `${dataset?.label || "LEPISZCZE"} model results, sorted by ${metricLabel(state.metric)} ${state.direction === "desc" ? "descending" : "ascending"}`;
    elements.empty.hidden = entries.length !== 0;

    const headingRow = document.createElement("tr");
    headingRow.innerHTML = "<th scope=\"col\">Rank</th><th scope=\"col\">Model</th>";
    metrics.forEach((metric) => {
      const th = document.createElement("th");
      th.scope = "col";
      if (metric === state.metric) {
        th.setAttribute("aria-sort", state.direction === "desc" ? "descending" : "ascending");
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = `metric-sort${metric === state.metric ? " is-sorted" : ""}`;
      button.dataset.sortMetric = metric;
      button.textContent = `${metricLabel(metric)}${metric === state.metric ? (state.direction === "desc" ? " ↓" : " ↑") : ""}`;
      button.setAttribute("aria-label", `Sort by ${metricLabel(metric)}`);
      th.append(button);
      headingRow.append(th);
    });
    elements.tableHead.replaceChildren(headingRow);

    const rows = entries.map((entry, index) => {
      const row = document.createElement("tr");
      const rank = document.createElement("td");
      rank.innerHTML = `<span class="rank-badge">${index + 1}</span>`;
      row.append(rank);

      const model = document.createElement("td");
      const parts = modelParts(entry.model);
      const modelName = document.createElement("span");
      modelName.className = "model-name";
      modelName.textContent = parts.name;
      modelName.title = entry.model;
      const owner = document.createElement("span");
      owner.className = "model-family";
      owner.textContent = parts.owner;
      model.append(modelName, owner);
      row.append(model);

      metrics.forEach((metric) => {
        const cell = document.createElement("td");
        const result = entry.metrics[metric];
        if (!result) {
          cell.textContent = "—";
        } else {
          const value = document.createElement("span");
          value.className = `metric-value${metric === state.metric ? " is-ranking" : ""}`;
          value.textContent = formatNumber(result.mean);
          cell.append(value);
          if (result.std !== null) {
            const std = document.createElement("span");
            std.className = "metric-std";
            std.textContent = `± ${formatNumber(result.std)}`;
            cell.append(std);
          }
        }
        row.append(cell);
      });
      return row;
    });
    elements.tableBody.replaceChildren(...rows);
  };

  const render = ({ resetDataset = false } = {}) => {
    refreshTaskOptions();
    if (resetDataset) state.dataset = "";
    refreshDatasetOptions();
    refreshMetricOptions();
    elements.search.value = state.search;
    renderTable();
    writeUrlState();
  };

  const exportRows = () => {
    const metrics = metricsForDataset();
    return rankedEntries().map((entry, index) => {
      const row = { rank: index + 1, model: entry.model, task: entry.task, dataset: entry.dataset };
      metrics.forEach((metric) => {
        row[metric] = entry.metrics[metric]?.mean ?? null;
        row[`${metric}_std`] = entry.metrics[metric]?.std ?? null;
      });
      return row;
    });
  };

  const download = (format) => {
    const rows = exportRows();
    let body;
    let type;
    if (format === "json") {
      body = `${JSON.stringify(rows, null, 2)}\n`;
      type = "application/json";
    } else {
      const headers = Object.keys(rows[0] || { rank: "", model: "", task: "", dataset: "" });
      const escape = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
      body = [headers.map(escape).join(","), ...rows.map((row) => headers.map((key) => escape(row[key])).join(","))].join("\n");
      type = "text/csv";
    }
    const url = URL.createObjectURL(new Blob([body], { type }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `lepiszcze-${state.dataset.replaceAll("/", "-")}.${format}`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const bindEvents = () => {
    elements.task.addEventListener("change", () => {
      state.task = elements.task.value;
      state.metric = "";
      state.direction = "desc";
      render({ resetDataset: true });
    });
    elements.dataset.addEventListener("change", () => {
      state.dataset = elements.dataset.value;
      state.metric = "";
      state.direction = "desc";
      render();
    });
    elements.metric.addEventListener("change", () => {
      state.metric = elements.metric.value;
      state.direction = "desc";
      render();
    });
    elements.search.addEventListener("input", () => {
      state.search = elements.search.value;
      renderTable();
      writeUrlState();
    });
    elements.tableHead.addEventListener("click", (event) => {
      const button = event.target.closest("[data-sort-metric]");
      if (!button) return;
      if (state.metric === button.dataset.sortMetric) {
        state.direction = state.direction === "desc" ? "asc" : "desc";
      } else {
        state.metric = button.dataset.sortMetric;
        state.direction = "desc";
      }
      elements.metric.value = state.metric;
      renderTable();
      writeUrlState();
    });
    elements.share.addEventListener("click", async () => {
      const previous = elements.share.textContent;
      try {
        await navigator.clipboard.writeText(window.location.href);
        elements.share.textContent = "Link copied";
      } catch {
        elements.share.textContent = "Copy the URL above";
      }
      window.setTimeout(() => { elements.share.textContent = previous; }, 1800);
    });
    panel.querySelectorAll("[data-export]").forEach((button) => {
      button.addEventListener("click", () => download(button.dataset.export));
    });
  };

  const initialise = async () => {
    try {
      const response = await fetch("../data/leaderboard.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.data = await response.json();
      if (state.data.schema_version !== 2 || !Array.isArray(state.data.entries)) {
        throw new Error("unsupported data schema");
      }
      readUrlState();
      document.querySelector("[data-summary='results']").textContent = state.data.summary.entries;
      document.querySelector("[data-summary='datasets']").textContent = state.data.summary.datasets;
      document.querySelector("[data-summary='models']").textContent = state.data.summary.models;
      elements.sourceLink.href = `${state.data.source.repository}/tree/${state.data.source.commit}/${state.data.source.results_path}`;
      bindEvents();
      render();
      elements.status.hidden = true;
      elements.content.hidden = false;
      panel.setAttribute("aria-busy", "false");
    } catch (error) {
      console.error("Unable to load leaderboard", error);
      elements.status.classList.add("is-error");
      elements.status.textContent = "The leaderboard data could not be loaded. Please refresh the page or inspect the source repository.";
      panel.setAttribute("aria-busy", "false");
    }
  };

  initialise();
})();
