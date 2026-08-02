import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const CUSTOM_MODEL = "⚪ — | use custom_model";

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
    if (node.__nvidiaSessionKeyWidget) return;

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
        const timeoutWidget = node.widgets?.find((widget) => widget.name === "timeout_seconds");
        status.textContent = "verifying…";
        try {
            const response = await api.fetchApi("/nvidia-nim-outfit-caption/session", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: String(node.id),
                    api_key: input.value,
                    api_key_env: envWidget?.value || "NVIDIA_API_KEY_COMFYUI",
                    timeout_seconds: Number(timeoutWidget?.value) || 15,
                }),
            });
            const payload = await response.json();
            const modelWidget = node.widgets?.find((widget) => widget.name === "model");
            const labels = (payload.models || []).map((item) => item.label);
            const values = labels.length ? labels : [CUSTOM_MODEL];
            if (modelWidget) {
                modelWidget.options = { ...(modelWidget.options || {}), values };
                if (!values.includes(modelWidget.value)) modelWidget.value = values[0];
                node.setDirtyCanvas(true, true);
            }
            const failures = Object.entries(payload.probe_errors || {});
            status.textContent = payload.error ? "stale/error" : payload.stale ? "stale" : `${labels.length} verified`;
            status.title = payload.error || failures.map(([id, error]) => `${id}: ${error}`).join("\n");
        } catch (error) {
            status.textContent = "refresh failed";
            status.title = String(error);
        }
    };
    button.addEventListener("click", refresh);
    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") refresh();
    });

    node.__nvidiaSessionKeyWidget = node.addDOMWidget("api_key_session", "custom", row, {
        serialize: false,
        getMinHeight: () => 30,
        getMaxHeight: () => 34,
    });
    queueMicrotask(refresh);
}

app.registerExtension({
    name: "Delcado.NvidiaNimOutfitCaption.Secret",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "NvidiaNimOutfitCaption") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            installReadableLabels(this);
            installSessionKeyWidget(this);
            return result;
        };
    },
});
