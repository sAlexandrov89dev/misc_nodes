import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "LoadImagesFromFolder",
    async nodeCreated(node) {
        if (node.comfyClass !== "LoadImagesFromFolder") return;

        const folderWidget = node.widgets.find(w => w.name === "folder");
        if (!folderWidget) return;

        node.addWidget("button", "select folder", null, async () => {
            const resp = await api.fetchApi("/select_folder");
            const data = await resp.json();
            if (data.path) folderWidget.value = data.path;
        });
    }
});