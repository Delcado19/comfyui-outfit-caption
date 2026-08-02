import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const READABLE_LABELS = {
    custom_model: "custom model",
    api_key_env: "API key environment variable",
    timeout_seconds: "timeout (seconds)",
    max_image_mb: "max. image size (MB)",
    max_tokens: "max. output tokens",
    no_person_accessories: "remove person accessories from VTON prompt",
};

function installReadableLabels(node) {
    for (const widget of node.widgets || []) {
        if (READABLE_LABELS[widget.name]) widget.label = READABLE_LABELS[widget.name];
    }
}

function installSessionKeyWidget(node) {
    if (node.__geminiSessionKeyWidget) return;

    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:6px;align-items:center;padding:4px 8px;";
    const input = document.createElement("input");
    input.type = "password";
    input.placeholder = "Session API key (not saved)";
    input.autocomplete = "off";
    input.style.cssText = "min-width:0;flex:1;background:#222;color:#ddd;border:1px solid #555;padding:4px;";
    const button = document.createElement("button");
    button.textContent = "Refresh models";
    const status = document.createElement("span");
    status.style.cssText = "font-size:10px;color:#aaa;white-space:nowrap;";
    row.append(input, button, status);

    const refresh = async () => {
        const envWidget = node.widgets?.find((widget) => widget.name === "api_key_env");
        status.textContent = "loading…";
        try {
            const response = await api.fetchApi("/gemini-outfit-caption/session", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: String(node.id),
                    api_key: input.value,
                    api_key_env: envWidget?.value || "GEMINI_API_KEY_COMFYUI",
                }),
            });
            const payload = await response.json();
            const modelWidget = node.widgets?.find((widget) => widget.name === "model");
            const labels = (payload.models || []).map((item) => item.label);
            if (modelWidget && labels.length) {
                modelWidget.options = { ...(modelWidget.options || {}), values: labels };
                if (!labels.includes(modelWidget.value)) modelWidget.value = labels[0];
                node.setDirtyCanvas(true, true);
            }
            status.textContent = payload.error ? "stale/error" : payload.stale ? "stale" : `${labels.length} live`;
            status.title = payload.error || "";
        } catch (error) {
            status.textContent = "refresh failed";
            status.title = String(error);
        }
    };
    button.addEventListener("click", refresh);
    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") refresh();
    });

    node.__geminiSessionKeyWidget = node.addDOMWidget("api_key_session", "custom", row, {
        serialize: false,
        getMinHeight: () => 30,
        getMaxHeight: () => 34,
    });
    queueMicrotask(refresh);
}

app.registerExtension({
    name: "Delcado.GeminiOutfitCaption.Secret",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeminiOutfitCaption") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            installReadableLabels(this);
            installSessionKeyWidget(this);
            return result;
        };
    },
});
