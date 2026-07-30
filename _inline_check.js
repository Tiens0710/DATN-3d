
        // Global API Configuration
        let backendUrl = "";
        let isConnected = false;
        const ngrokInput = document.getElementById("ngrok-url");
        ngrokInput.value = localStorage.getItem("datn-ngrok-url") || "";
        ngrokInput.addEventListener("change", () => {
            localStorage.setItem("datn-ngrok-url", ngrokInput.value.trim());
        });



        // SpatialFlow shell interactions and fixed 3D viewport.
        let spatialLatestSceneUrl = "";
        let spatialLatestZipUrl = "";

        function syncSpatialViewport(objectUrl, sceneUrl, zipUrl) {
            const globalViewer = document.getElementById("spatial-global-viewer");
            const emptyState = document.getElementById("spatial-viewport-empty");
            const stateLabel = document.getElementById("spatial-viewport-state");
            if (!globalViewer || !objectUrl) return;

            spatialLatestSceneUrl = sceneUrl || "";
            spatialLatestZipUrl = zipUrl || "";
            globalViewer.src = objectUrl;
            globalViewer.hidden = false;
            if (emptyState) emptyState.style.display = "none";
            if (stateLabel) stateLabel.textContent = "Scene ready";
        }

        function setSpatialViewportError(message) {
            const stateLabel = document.getElementById("spatial-viewport-state");
            if (stateLabel) stateLabel.textContent = message || "Preview failed";
        }

        const nodeLibraryDrawer = document.getElementById("node-library-drawer");
        const executionDrawer = document.getElementById("execution-drawer");
        const toggleDrawer = (drawer) => {
            if (!drawer) return;
            const willOpen = !drawer.classList.contains("open");
            nodeLibraryDrawer?.classList.remove("open");
            executionDrawer?.classList.remove("open");
            if (willOpen) drawer.classList.add("open");
        };

        document.getElementById("btn-toggle-library")?.addEventListener("click", () => toggleDrawer(nodeLibraryDrawer));
        document.getElementById("btn-panel-library")?.addEventListener("click", () => toggleDrawer(nodeLibraryDrawer));
        document.getElementById("btn-toggle-logs")?.addEventListener("click", () => toggleDrawer(executionDrawer));
        document.getElementById("btn-panel-logs")?.addEventListener("click", () => toggleDrawer(executionDrawer));

        const spatialViewer = document.getElementById("spatial-global-viewer");
        document.getElementById("btn-view-spin")?.addEventListener("click", (event) => {
            if (!spatialViewer) return;
            if (spatialViewer.hasAttribute("auto-rotate")) {
                spatialViewer.removeAttribute("auto-rotate");
                event.currentTarget.textContent = "⟳ Auto rotate";
            } else {
                spatialViewer.setAttribute("auto-rotate", "");
                event.currentTarget.textContent = "⏸ Stop rotate";
            }
        });
        document.getElementById("btn-view-reset")?.addEventListener("click", () => {
            if (!spatialViewer) return;
            spatialViewer.cameraOrbit = "0deg 75deg auto";
            spatialViewer.fieldOfView = "auto";
        });
        document.getElementById("btn-view-front")?.addEventListener("click", () => {
            if (spatialViewer) spatialViewer.cameraOrbit = "0deg 75deg auto";
        });
        document.getElementById("btn-view-top")?.addEventListener("click", () => {
            if (spatialViewer) spatialViewer.cameraOrbit = "0deg 5deg auto";
        });
        document.getElementById("btn-view-side")?.addEventListener("click", () => {
            if (spatialViewer) spatialViewer.cameraOrbit = "90deg 75deg auto";
        });
        document.getElementById("btn-spatial-download")?.addEventListener("click", async () => {
            if (!spatialLatestSceneUrl) {
                writeLog("Chưa có mô hình GLB để tải.", "error");
                return;
            }
            try {
                await downloadBackendFile(spatialLatestSceneUrl, "scene_combined.glb");
            } catch (error) {
                writeLog(`Lỗi tải scene GLB: ${error.message}`, "error");
            }
        });

        document.querySelectorAll(".render-segment").forEach((button) => {
            button.addEventListener("click", () => {
                document.querySelectorAll(".render-segment").forEach((item) => item.classList.remove("active"));
                button.classList.add("active");
                if (button.textContent.trim() === "Wireframe") {
                    writeLog("Model Viewer không hỗ trợ wireframe thật; đang giữ chế độ textured.", "info");
                }
            });
        });


        // Output logs styling and helper
        function writeLog(message, type = "info") {
            const logContent = document.getElementById("log-content");
            const time = new Date().toLocaleTimeString();
            let color = "#e4e4e7"; // default zinc

            if (type === "error") color = "#f87171"; // red
            if (type === "warning") color = "#fbbf24"; // amber
            if (type === "system") color = "#60a5fa"; // blue
            if (type === "success") color = "#34d399"; // emerald

            logContent.innerHTML += `<div style="color: ${color}; margin-bottom: 4px;">[${time}] ${message}</div>`;
            logContent.scrollTop = logContent.scrollHeight;
        }

        async function downloadBackendFile(fileUrl, filename) {
            const url = new URL(fileUrl, backendUrl || window.location.href);
            url.searchParams.set("ngrok-skip-browser-warning", "1");
            const response = await fetch(url.toString(), {
                headers: { "ngrok-skip-browser-warning": "1" }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const blob = await response.blob();
            const objectUrl = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = objectUrl;
            anchor.download = filename;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
        }

        async function downloadBackendFiles(files) {
            for (const file of files) {
                await downloadBackendFile(file.url, file.filename);
                await new Promise(resolve => setTimeout(resolve, 250));
            }
        }

        document.getElementById("btn-copy-logs").onclick = () => {
            const logContent = document.getElementById("log-content");
            const logsText = logContent.innerText || logContent.textContent;
            
            navigator.clipboard.writeText(logsText).then(() => {
                const copyBtn = document.getElementById("btn-copy-logs");
                const originalText = copyBtn.innerText;
                copyBtn.innerText = "COPIED!";
                copyBtn.style.color = "#34d399"; // Green success color
                setTimeout(() => {
                    copyBtn.innerText = originalText;
                    copyBtn.style.color = ""; // Reset
                }, 1500);
            }).catch(err => {
                writeLog(`❌ Không thể sao chép log: ${err.message}`, "error");
            });
        };

        document.getElementById("btn-clear-logs").onclick = () => {
            document.getElementById("log-content").innerHTML = "";
        };

        // Test API connection
        document.getElementById("btn-connect").onclick = async () => {
            let url = document.getElementById("ngrok-url").value.trim();
            if (!url) {
                writeLog("Vui lòng nhập đường dẫn Ngrok!", "error");
                return;
            }
            if (url.endsWith("/")) url = url.slice(0, -1);

            writeLog(`Đang kết nối tới server GPU Kaggle: ${url}...`, "system");
            try {
                const res = await fetch(`${url}/api/health`, {
                    method: 'GET',
                    headers: { "ngrok-skip-browser-warning": "69420" }
                });
                if (res.ok) {
                    const data = await res.json();
                    backendUrl = url;
                    isConnected = true;
                    document.getElementById("status-indicator").classList.add("connected");
                    document.getElementById("btn-connect").innerText = "Connected";
                    localStorage.setItem("datn-ngrok-url", url);
                    writeLog(`✅ Đã kết nối thành công! Phiên bản: ${data.version || '1.0.0'}`, "success");
                } else {
                    throw new Error("Mã trạng thái phản hồi lỗi.");
                }
            } catch (e) {
                isConnected = false;
                document.getElementById("status-indicator").classList.remove("connected");
                document.getElementById("btn-connect").innerText = "Connect";
                writeLog(`❌ Lỗi kết nối: ${e.message}. Hãy đảm bảo server trên Kaggle đang chạy và Ngrok đã kết nối thành công.`, "error");
            }
        };

        // Initialize LiteGraph
        const graph = new LGraph();
        const canvas = new LGraphCanvas("#graphcanvas", graph);
        canvas.show_stats = false; // Hide default performance statistics
        canvas.background_image = null; // Background handled by CSS grid pattern

        // =========================================================================
        // COMFYUI-AUTHENTIC THEME
        // =========================================================================
        LiteGraph.NODE_DEFAULT_BGCOLOR = "#26282b";
        LiteGraph.NODE_DEFAULT_BOXCOLOR = "#111214";
        LiteGraph.NODE_TEXT_COLOR = "#e6e6e6";
        LiteGraph.NODE_TITLE_COLOR = "#f2f2f2";
        LiteGraph.NODE_SELECTED_TITLE_COLOR = "#ffffff";
        LiteGraph.NODE_DEFAULT_SHAPE = "box";
        LiteGraph.LINK_COLOR = "#8e959e";
        LiteGraph.CONNECTING_LINK_COLOR = "#ffffff";
        LiteGraph.DEFAULT_LINK_TYPE_WIDTH = 2;
        LiteGraph.CANVAS_GRID_SIZE = 24;
        LiteGraph.NODE_TITLE_HEIGHT = 26;
        LiteGraph.NODE_SLOT_HEIGHT = 20;

        // ComfyUI fonts
        LiteGraph.NODE_TITLE_TEXT_Y = 18;
        LiteGraph.NODE_TEXT_SIZE = 12;
        LiteGraph.NODE_SUBTEXT_SIZE = 10;
        LiteGraph.NODE_TITLE_FONT = "600 12px Inter, 'Segoe UI', sans-serif";
        LiteGraph.NODE_LABEL_FONT = "normal 11px Inter, 'Segoe UI', sans-serif";

        // ComfyUI wire colors by slot type
        LGraphCanvas.link_type_colors = {
            "string": "#7ecb72",
            "object": "#64a8e8",
            "number": "#d8a95d",
            "boolean": "#d981b5"
        };

        // ComfyUI-style smooth curved links
        canvas.render_connections_border = false;
        canvas.connections_width = 2;
        canvas.render_curved_connections = true;
        canvas.render_connection_arrows = false;
        canvas.highquality_render = true;
        canvas.links_render_mode = LiteGraph.SPLINE_LINK;

        // Redraw loop
        function resizeCanvas() {
            const parent = document.getElementById("canvas-container");
            if (parent) {
                canvas.resize(parent.clientWidth, parent.clientHeight);
                canvas.draw(true, true);
            }
        }
        window.addEventListener("resize", resizeCanvas);
        window.addEventListener("load", resizeCanvas);
        setTimeout(resizeCanvas, 100);

        // =========================================================================
        // CUSTOM LITEGRAPH NODES DEFINITION
        // =========================================================================

        // Node 1: Input Text Prompt
        function NodePrompt() {
            this.addOutput("prompt", "string");
            this.properties = {
                prompt: "a wooden dining chair next to a small wooden side table, product photography, plain dark background, studio lighting"
            };
            this.addWidget("button", "▶ Run to here", () => {
                if (window.runPipelineUntil) window.runPipelineUntil(this);
            });
            this.size = [340, 180];
            this.color = "#5b7a3a";
            this.bgcolor = "#26282b";
        }
        NodePrompt.title = "✏️ Input Text Prompt";
        NodePrompt.title_color = "rgba(76, 175, 80, 0.85)";

        NodePrompt.prototype.onExecute = function () {
            const textarea = document.getElementById(`prompt-input-${this.id}`);
            if (textarea) {
                this.properties.prompt = textarea.value;
            }
            this.setOutputData(0, this.properties.prompt);
        };

        NodePrompt.prototype.updateTextarea = function () {
            const textareaId = `prompt-input-${this.id}`;
            let textarea = document.getElementById(textareaId);

            if (!textarea) {
                textarea = document.createElement("textarea");
                textarea.id = textareaId;
                textarea.className = "node-textarea";
                textarea.value = this.properties.prompt;
                textarea.placeholder = "Nhập mô tả tiếng Anh tại đây...";

                textarea.addEventListener("input", () => {
                    this.properties.prompt = textarea.value;
                });

                document.getElementById("canvas-container").appendChild(textarea);
            }

            const scale = canvas.ds.scale;
            const screenPos = canvas.convertOffsetToCanvas([this.pos[0] + 15, this.pos[1] + 45]);
            const x = screenPos[0];
            const y = screenPos[1];
            const w = (this.size[0] - 30) * scale;
            const h = (this.size[1] - 60) * scale;

            textarea.style.left = `${x}px`;
            textarea.style.top = `${y}px`;
            textarea.style.width = `${w}px`;
            textarea.style.height = `${h}px`;
            textarea.style.fontSize = `${Math.max(9, 13 * scale)}px`;
            textarea.style.padding = `${Math.max(3, 9 * scale)}px`;
            textarea.style.lineHeight = "1.35";

            if (scale < 0.45) {
                textarea.style.display = "none";
            } else {
                textarea.style.display = "block";
            }
        };

        NodePrompt.prototype.onDrawForeground = function (ctx) {
            if (canvas.ds.scale >= 0.45) return;

            const prompt = this.properties.prompt || "";
            const x = 15;
            const y = 45;
            const width = this.size[0] - 30;
            const height = this.size[1] - 60;

            ctx.save();
            ctx.fillStyle = "#161719";
            ctx.strokeStyle = "#3b3d41";
            ctx.lineWidth = 1;
            ctx.fillRect(x, y, width, height);
            ctx.strokeRect(x, y, width, height);

            ctx.beginPath();
            ctx.rect(x + 6, y + 5, width - 12, height - 10);
            ctx.clip();

            ctx.fillStyle = "#e7e7e7";
            ctx.font = "12px Inter, 'Segoe UI', sans-serif";

            const words = prompt.split(/\s+/);
            const lines = [];
            let line = "";

            for (const word of words) {
                const candidate = line ? `${line} ${word}` : word;
                if (ctx.measureText(candidate).width > width - 20) {
                    if (line) lines.push(line);
                    line = word;
                } else {
                    line = candidate;
                }
            }

            if (line) lines.push(line);

            lines.slice(0, 4).forEach((text, index) => {
                ctx.fillText(text, x + 10, y + 20 + index * 16);
            });

            if (lines.length > 4) {
                ctx.fillText("…", x + 10, y + height - 8);
            }

            ctx.restore();
        };

        NodePrompt.prototype.onRemoved = function () {
            const textareaId = `prompt-input-${this.id}`;
            const textarea = document.getElementById(textareaId);
            if (textarea) textarea.remove();
        };

        LiteGraph.registerNodeType("datn/input_prompt", NodePrompt);


        // Node 2: Scene Graph Parser (spaCy NLP)
        function NodeSceneGraph() {
            this.addInput("prompt", "string");
            this.addOutput("scene_graph", "object");
            this.properties = { graphData: null };
            this.addWidget("button", "▶ Run to here", () => {
                if (window.runPipelineUntil) window.runPipelineUntil(this);
            });
            this.size = [340, 200];
            this.color = "#2e7d32";
            this.bgcolor = "#26282b";
        }
        NodeSceneGraph.title = "🧠 Scene Graph Parser";
        NodeSceneGraph.title_color = "rgba(46, 125, 50, 0.85)";
        NodeSceneGraph.prototype.onExecute = async function () {
            const inputPrompt = this.getInputData(0);
            if (!inputPrompt) {
                writeLog("❌ Scene Graph Parser: Thiếu dữ liệu đầu vào prompt!", "error");
                throw new Error("Thiếu dữ liệu đầu vào prompt");
            }
            if (!isConnected) {
                writeLog("❌ Scene Graph Parser: Chưa kết nối GPU Kaggle!", "error");
                throw new Error("Chưa kết nối GPU Kaggle");
            }

            if (this._lastPrompt === inputPrompt && this.properties.graphData) {
                this.setOutputData(0, this.properties.graphData);
                return;
            }

            this._lastPrompt = inputPrompt;
            writeLog("Gửi prompt phân tích quan hệ cú pháp không gian (Scene Graph)...", "system");

            try {
                const res = await fetch(`${backendUrl}/api/parse_scene_graph`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "ngrok-skip-browser-warning": "69420"
                    },
                    body: JSON.stringify({ text: inputPrompt })
                });
                if (res.ok) {
                    const data = await res.json();
                    this.properties.graphData = data;
                    this.setOutputData(0, data);
                    writeLog("✅ Trích xuất Scene Graph thành công!", "success");
                } else {
                    const errText = await res.text();
                    throw new Error(`API error: ${res.status} - ${errText}`);
                }
            } catch (e) {
                writeLog(`❌ Lỗi Scene Graph: ${e.message}`, "error");
                throw e;
            }
        };
        NodeSceneGraph.prototype.onDrawForeground = function (ctx) {
            if (!this.properties.graphData) {
                ctx.fillStyle = "#52525b";
                ctx.font = "italic 13px 'Outfit'";
                ctx.fillText("Chờ dữ liệu prompt...", 20, 60);
                return;
            }

            ctx.fillStyle = "#818cf8";
            ctx.font = "bold 13px 'Outfit'";
            ctx.fillText("Vật thể phát hiện (Nodes):", 15, 55);

            const nodes = this.properties.graphData.nodes || [];
            nodes.forEach((n, idx) => {
                ctx.fillStyle = "#a5b4fc";
                ctx.font = "12px 'Fira Code'";
                ctx.fillText(`• [${n.id}] ${n.label} (${n.full})`, 25, 78 + (idx * 20));
            });

            const edges = this.properties.graphData.edges || [];
            if (edges.length > 0) {
                ctx.fillStyle = "#ec4899";
                ctx.font = "bold 13px 'Outfit'";
                ctx.fillText("Quan hệ không gian (Edges):", 15, 78 + (nodes.length * 20) + 12);

                edges.forEach((e, idx) => {
                    ctx.fillStyle = "#f472b6";
                    ctx.font = "12px 'Fira Code'";
                    ctx.fillText(`• ${e.subject} --[${e.relation}]--> ${e.object}`, 25, 78 + (nodes.length * 20) + 30 + (idx * 20));
                });
            }
        };
        LiteGraph.registerNodeType("datn/scene_graph", NodeSceneGraph);


        // Node 3: 2D Layout Generator (Bounding Box layout rules)
        function NodeLayout2D() {
            this.addInput("scene_graph", "object");
            this.addOutput("layout", "object");
            this.properties = { layoutData: null };
            this.addWidget("button", "▶ Run to here", () => {
                if (window.runPipelineUntil) window.runPipelineUntil(this);
            });
            this.size = [380, 260];
            this.color = "#6a4dc7";
            this.bgcolor = "#26282b";
        }
        NodeLayout2D.title = "📐 2D Layout Generator";
        NodeLayout2D.title_color = "rgba(106, 77, 199, 0.85)";
        NodeLayout2D.prototype.onExecute = async function () {
            const graphData = this.getInputData(0);
            if (!graphData) {
                writeLog("❌ 2D Layout Generator: Thiếu dữ liệu đầu vào scene_graph!", "error");
                throw new Error("Thiếu dữ liệu đầu vào scene_graph");
            }
            if (!isConnected) {
                writeLog("❌ 2D Layout Generator: Chưa kết nối GPU Kaggle!", "error");
                throw new Error("Chưa kết nối GPU Kaggle");
            }

            if (this._lastGraph === graphData && this.properties.layoutData) {
                this.setOutputData(0, this.properties.layoutData);
                return;
            }

            this._lastGraph = graphData;
            writeLog("Tính toán bố cục lưới Bounding Box 2D...", "system");

            try {
                const res = await fetch(`${backendUrl}/api/generate_layout`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "ngrok-skip-browser-warning": "69420"
                    },
                    body: JSON.stringify(graphData)
                });
                if (res.ok) {
                    const data = await res.json();
                    this.properties.layoutData = data;
                    this.setOutputData(0, data);
                    writeLog("✅ Đã sinh tọa độ bố cục 2D.", "success");
                } else {
                    const errText = await res.text();
                    throw new Error(`API error: ${res.status} - ${errText}`);
                }
            } catch (e) {
                writeLog(`❌ Lỗi tính toán Layout: ${e.message}`, "error");
                throw e;
            }
        };
        NodeLayout2D.prototype.onDrawForeground = function (ctx) {
            if (!this.properties.layoutData) {
                ctx.fillStyle = "#52525b";
                ctx.font = "italic 13px 'Outfit'";
                ctx.fillText("Chờ cấu trúc Graph...", 20, 60);
                return;
            }

            const boxes = this.properties.layoutData.layout || {};

            // Render canvas container inside node (Kích thước lớn hơn: 200x200)
            ctx.save();
            ctx.fillStyle = "#09090b";
            ctx.fillRect(15, 45, 200, 200);
            ctx.strokeStyle = "#27272a";
            ctx.lineWidth = 1.5;
            ctx.strokeRect(15, 45, 200, 200);

            // Draw grid lines inside preview
            ctx.strokeStyle = "rgba(255,255,255,0.02)";
            ctx.lineWidth = 1;
            for (let i = 1; i < 4; i++) {
                ctx.beginPath();
                ctx.moveTo(15 + (i * 50), 45);
                ctx.lineTo(15 + (i * 50), 245);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(15, 45 + (i * 50));
                ctx.lineTo(215, 45 + (i * 50));
                ctx.stroke();
            }

            const colors = ["#818cf8", "#34d399", "#fb7185", "#fbbf24"];
            let colorIdx = 0;

            for (const [nid, box] of Object.entries(boxes)) {
                const scale = 200 / 512;
                const bx = 15 + (box.x * scale);
                const by = 45 + (box.y * scale);
                const bw = box.w * scale;
                const bh = box.h * scale;

                const color = colors[colorIdx % colors.length];
                ctx.fillStyle = color + "20"; // 12% opacity fill
                ctx.fillRect(bx, by, bw, bh);
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(bx, by, bw, bh);

                ctx.fillStyle = color;
                ctx.font = "bold 10px 'Outfit'";
                ctx.fillText(nid, bx + 4, by + 14);
                colorIdx++;
            }

            ctx.fillStyle = "#e4e4e7";
            ctx.font = "bold 13px 'Outfit'";
            ctx.fillText("Bố cục Bounding Box (px):", 230, 55);
            let textY = 75;
            for (const [nid, box] of Object.entries(boxes)) {
                ctx.fillStyle = colors[(colorIdx - 1) % colors.length];
                ctx.font = "12px 'Fira Code'";
                ctx.fillText(`${nid}:`, 230, textY);
                ctx.fillStyle = "#a1a1aa";
                ctx.fillText(`x=${box.x}, y=${box.y}`, 240, textY + 16);
                ctx.fillText(`w=${box.w}, h=${box.h}`, 240, textY + 30);
                textY += 48;
            }
            ctx.restore();
        };
        LiteGraph.registerNodeType("datn/layout_2d", NodeLayout2D);


        // Node 4: Gemini prompt optimizer
        function NodePromptOptimizer() {
            this.addInput("prompt", "string");
            this.addOutput("optimized_prompt", "string");
            this.properties = { optimizedPrompt: "" };
            this.addWidget("button", "▶ Run to here", () => {
                if (window.runPipelineUntil) window.runPipelineUntil(this);
            });
            this.size = [380, 210];
            this.color = "#31558f";
            this.bgcolor = "#26282b";
        }
        NodePromptOptimizer.title = "Gemini 3.1 Flash-Lite Prompt Optimizer";
        NodePromptOptimizer.title_color = "#ffffff";
        NodePromptOptimizer.prototype.onExecute = async function () {
            const prompt = this.getInputData(0);
            if (!prompt) throw new Error("Thiếu prompt cho Gemini Optimizer");
            if (!isConnected) throw new Error("Chưa kết nối GPU Kaggle");

            if (this._lastPrompt === prompt && this.properties.optimizedPrompt) {
                this.setOutputData(0, this.properties.optimizedPrompt);
                return;
            }

            this._lastPrompt = prompt;
            writeLog("Đang tối ưu prompt bằng Gemini 3.1 Flash-Lite...", "system");

            const res = await fetch(`${backendUrl}/api/optimize_prompt`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "69420"
                },
                body: JSON.stringify({ prompt })
            });

            if (!res.ok) {
                throw new Error(`Gemini API error: ${res.status} - ${await res.text()}`);
            }

            const data = await res.json();
            this.properties.optimizedPrompt = data.optimized_prompt || prompt;
            this.setOutputData(0, this.properties.optimizedPrompt);
            writeLog(
                data.used_gemini
                    ? "✅ Gemini đã tối ưu prompt."
                    : "⚠️ Gemini fallback: tiếp tục với prompt đã chuẩn hóa.",
                data.used_gemini ? "success" : "warning"
            );
            this.setDirtyCanvas(true, true);
        };
        NodePromptOptimizer.prototype.onDrawForeground = function (ctx) {
            ctx.save();
            ctx.fillStyle = "#d7e3ff";
            ctx.font = "11px Inter, 'Segoe UI', sans-serif";

            const text = this.properties.optimizedPrompt ||
                "Chờ tối ưu prompt...";
            const maxWidth = this.size[0] - 30;
            const lines = [];
            let line = "";

            for (const word of text.split(/\s+/)) {
                const candidate = line ? `${line} ${word}` : word;
                if (ctx.measureText(candidate).width > maxWidth) {
                    if (line) lines.push(line);
                    line = word;
                } else {
                    line = candidate;
                }
            }
            if (line) lines.push(line);

            lines.slice(0, 7).forEach((item, index) => {
                ctx.fillText(item, 15, 72 + index * 16);
            });

            ctx.restore();
        };
        LiteGraph.registerNodeType("datn/prompt_optimizer", NodePromptOptimizer);


        // Node 5: SD3.5 Image Generator
        function NodeSDXLGen() {
            this.addInput("prompt", "string");
            this.addOutput("image_url", "string");

            this.properties = {
                imageUrl: ""
            };
            this.addWidget("button", "▶ Run to here", () => {
                if (window.runPipelineUntil) window.runPipelineUntil(this);
            });
            this.addWidget("button", "Download PNG", async () => {
                if (!this.properties.imageUrl) {
                    writeLog("Chưa có ảnh 2D để tải.", "warning");
                    return;
                }
                try {
                    await downloadBackendFile(this.properties.imageUrl, "generated_2d.png");
                } catch (error) {
                    writeLog(`Lỗi tải ảnh: ${error.message}`, "error");
                }
            });

            this.size = [450, 420];
            this.color = "rgba(198, 40, 40, 0.85)";
            this.bgcolor = "#26282b";
        }
        NodeSDXLGen.title = "🎨 SD3.5 Image Generator";
        NodeSDXLGen.title_color = "rgba(198, 40, 40, 0.85)";
        NodeSDXLGen.prototype.onExecute = async function () {
            const prompt = this.getInputData(0);
            if (!prompt) {
                writeLog("❌ SD3.5 Image Generator: Thiếu dữ liệu prompt!", "error");
                throw new Error("Thiếu dữ liệu prompt");
            }
            if (!isConnected) {
                writeLog("❌ SD3.5 Image Generator: Chưa kết nối GPU Kaggle!", "error");
                throw new Error("Chưa kết nối GPU Kaggle");
            }

            if (this._lastPrompt === prompt && this.properties.imageUrl) {
                this.setOutputData(0, this.properties.imageUrl);
                return;
            }

            this._lastPrompt = prompt;
            try {
                let generationPrompt = prompt;
                writeLog(
                    "Đang tối ưu prompt bằng Gemini 3.1 Flash-Lite...",
                    "system"
                );

                try {
                    const optimizeRes = await fetch(
                        `${backendUrl}/api/optimize_prompt`,
                        {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json",
                                "ngrok-skip-browser-warning": "69420"
                            },
                            body: JSON.stringify({ prompt })
                        }
                    );

                    if (optimizeRes.ok) {
                        const optimizeData = await optimizeRes.json();
                        generationPrompt =
                            optimizeData.optimized_prompt || prompt;
                        this.properties.optimizedPrompt = generationPrompt;

                        if (optimizeData.used_gemini) {
                            writeLog(
                                "✅ Gemini đã tối ưu prompt cho SD3.5.",
                                "success"
                            );
                        } else {
                            writeLog(
                                `⚠️ Tiếp tục với prompt gốc: ${optimizeData.warning || "Gemini chưa được cấu hình"}`,
                                "warning"
                            );
                        }
                    } else {
                        const optimizeError = await optimizeRes.text();
                        writeLog(
                            `⚠️ Gemini API lỗi, tiếp tục với prompt gốc: ${optimizeError}`,
                            "warning"
                        );
                    }
                } catch (optimizeError) {
                    writeLog(
                        `⚠️ Không kết nối được Gemini, tiếp tục với prompt gốc: ${optimizeError.message}`,
                        "warning"
                    );
                }

                writeLog(
                    "Chạy mô hình khuếch tán SD3.5 + LoRA...",
                    "system"
                );

                const res = await fetch(`${backendUrl}/api/generate_image`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "ngrok-skip-browser-warning": "69420"
                    },
                    body: JSON.stringify({
                        prompt: generationPrompt,
                        layout: {}, // Send empty layout to satisfy backend schema
                        lora_scale: 0.0
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    this.properties.imageUrl = `${backendUrl}${data.image_url}`;
                    this._imageObj = null; // Clear cached image to force reload
                    this.setOutputData(0, this.properties.imageUrl);
                    writeLog("✅ Đã sinh ảnh 2D kết quả thành công!", "success");
                    this.setDirtyCanvas(true, true);
                } else {
                    const errText = await res.text();
                    throw new Error(`API error: ${res.status} - ${errText}`);
                }
            } catch (e) {
                writeLog(`❌ Lỗi sinh ảnh SD3.5: ${e.message}`, "error");
                throw e;
            }
        };
        // The optimizer is now a separate node. Override the legacy handler
        // above so SD3.5 receives the optimizer output exactly once.
        NodeSDXLGen.prototype.onExecute = async function () {
            const prompt = this.getInputData(0);
            if (!prompt) throw new Error("Thiếu prompt cho SD3.5");
            if (!isConnected) throw new Error("Chưa kết nối GPU Kaggle");

            if (this._lastPrompt === prompt && this.properties.imageUrl) {
                this.setOutputData(0, this.properties.imageUrl);
                return;
            }

            this._lastPrompt = prompt;
            writeLog("Chạy mô hình khuếch tán SD3.5 mặc định...", "system");

            const res = await fetch(`${backendUrl}/api/generate_image`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "69420"
                },
                body: JSON.stringify({
                    prompt,
                    layout: {},
                    lora_scale: 0.0
                })
            });

            if (!res.ok) {
                throw new Error(`API error: ${res.status} - ${await res.text()}`);
            }

            const data = await res.json();
            this.properties.imageUrl = `${backendUrl}${data.image_url}`;
            this._imageObj = null;
            this.setOutputData(0, this.properties.imageUrl);
            writeLog("✅ Đã sinh ảnh 2D thành công.", "success");
            this.setDirtyCanvas(true, true);
        };

        NodeSDXLGen.prototype.onDrawBackground = function (ctx) {
            if (!this.properties.imageUrl) {
                ctx.fillStyle = "#52525b";
                ctx.font = "italic 13px 'Outfit'";
                ctx.fillText("Chờ xử lý pipeline...", 20, 95);
                return;
            }

            if (!this._imageObj || this._imageObj._rawUrl !== this.properties.imageUrl) {
                this._imageObj = new Image();
                this._imageObj._rawUrl = this.properties.imageUrl;
                fetch(this.properties.imageUrl, {
                    headers: { "ngrok-skip-browser-warning": "69420" }
                })
                .then(res => res.blob())
                .then(blob => {
                    this._imageObj.src = URL.createObjectURL(blob);
                    this._imageObj.onload = () => this.setDirtyCanvas(true, true);
                })
                .catch(e => console.error("Error loading SDXL preview:", e));
            }

            try {
                // Draw image preview inside node (Kích thước lớn hơn: 410x285)
                ctx.drawImage(this._imageObj, 15, 80, 410, 285);
                ctx.strokeStyle = "rgba(255,255,255,0.08)";
                ctx.lineWidth = 1;
                ctx.strokeRect(15, 80, 410, 285);
            } catch (e) { }
        };
        LiteGraph.registerNodeType("datn/sdxl_gen", NodeSDXLGen);


        // Node 5: Grounded-SAM2 segment extractor
        function NodeGroundedSAM2() {
            this.addInput("image_url", "string");
            this.addInput("layout", "object");
            this.addInput("prompt", "string");
            this.addOutput("crops_data", "object");

            this.properties = { 
                crops: [],
                labels: ""
            };
            this.addWidget("button", "▶ Run to here", () => {
                if (window.runPipelineUntil) window.runPipelineUntil(this);
            });
            this.addWidget("button", "Download SAM crops", async () => {
                const files = (this.properties.crops || [])
                    .filter(crop => crop.crop_url)
                    .map((crop, index) => ({
                        url: `${backendUrl}${crop.crop_url}`,
                        filename: `${crop.name || `object_${index + 1}`}.png`
                    }));
                if (!files.length) {
                    writeLog("Chưa có crop SAM2 để tải.", "warning");
                    return;
                }
                try {
                    await downloadBackendFiles(files);
                } catch (error) {
                    writeLog(`Lỗi tải crop SAM2: ${error.message}`, "error");
                }
            });
            this.addWidget("text", "Labels to Segment", this.properties.labels, (v) => {
                this.properties.labels = v;
            });
            this.size = [450, 420];
            this.color = "rgba(230, 81, 0, 0.85)";
            this.bgcolor = "#26282b";
        }
        NodeGroundedSAM2.title = "✂️ Grounded-SAM2 Segment";
        NodeGroundedSAM2.title_color = "rgba(230, 81, 0, 0.85)";
        NodeGroundedSAM2.prototype.onExecute = async function () {
            const imageUrl = this.getInputData(0);
            const inputLayout = this.getInputData(1);
            const userPrompt = this.getInputData(2) || "";
            if (!imageUrl) {
                writeLog("❌ Grounded-SAM2 Segment: Thiếu dữ liệu hình ảnh image_url!", "error");
                throw new Error("Thiếu dữ liệu hình ảnh image_url");
            }
            if (!isConnected) {
                writeLog("❌ Grounded-SAM2 Segment: Chưa kết nối GPU Kaggle!", "error");
                throw new Error("Chưa kết nối GPU Kaggle");
            }

            // Build layout object based on layout input or labels input widget
            let finalLayout = { layout: {} };
            if (inputLayout && inputLayout.layout && Object.keys(inputLayout.layout).length > 0) {
                finalLayout = inputLayout;
                // Extract active labels from layout keys
                const activeLabels = Object.keys(inputLayout.layout).map(nid => {
                    return nid.split("_")[0];
                }).filter(Boolean);
                this.properties.labels = activeLabels.join(", ");
                // Update widget text if visible
                if (this.widgets) {
                    const textWidget = this.widgets.find(w => w.name === "Labels to Segment");
                    if (textWidget) textWidget.value = this.properties.labels;
                }
            } else {
                const labelsArray = this.properties.labels.split(",").map(s => s.trim()).filter(Boolean);
                labelsArray.forEach((lbl, idx) => {
                    finalLayout.layout[`${lbl}_${idx}`] = { x: 0, y: 0, w: 0, h: 0 };
                });
            }

            const segmentationKey = JSON.stringify({
                imageUrl,
                userPrompt,
                layout: finalLayout,
                labels: this.properties.labels
            });

            if (
                this._lastSegmentationKey === segmentationKey &&
                this.properties.crops.length > 0
            ) {
                this.setOutputData(0, this.properties.crops);
                return;
            }

            this._lastSegmentationKey = segmentationKey;
            writeLog("Chạy GroundingDINO nhận diện & SAM2 cắt tách nền vật thể...", "system");

            try {
                const res = await fetch(`${backendUrl}/api/run_sam2`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "ngrok-skip-browser-warning": "69420"
                    },
                    body: JSON.stringify({
                        image_url: imageUrl,
                        layout: finalLayout,
                        prompt: userPrompt
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    this.properties.crops = data.crops || [];
                    this.setOutputData(0, this.properties.crops);
                    writeLog(`✅ SAM2 trích xuất thành công ${this.properties.crops.length} vật thể RGBA!`, "success");

                    this._cropImages = [];
                    this.setDirtyCanvas(true, true);
                } else {
                    const errText = await res.text();
                    throw new Error(`API error: ${res.status} - ${errText}`);
                }
            } catch (e) {
                writeLog(`❌ Lỗi Grounded-SAM2: ${e.message}`, "error");
                throw e;
            }
        };
        NodeGroundedSAM2.prototype.onDrawBackground = function (ctx) {
            const imageUrl = this.getInputData(0);
            if (!imageUrl) {
                ctx.fillStyle = "#52525b";
                ctx.font = "italic 13px 'Outfit'";
                ctx.fillText("Chờ nhận diện ảnh...", 20, 80);
                return;
            }

            // 1. Draw input image (cached)
            let previewUrl = imageUrl;
            if (this.properties.crops && this.properties.crops.length > 0) {
                previewUrl = `${backendUrl}/crops/sam2_visual.png`;
            }

            if (!this._inputImageObj || this._inputImageObj._rawUrl !== previewUrl) {
                this._inputImageObj = new Image();
                this._inputImageObj._rawUrl = previewUrl;
                fetch(previewUrl, {
                    headers: { "ngrok-skip-browser-warning": "69420" }
                })
                .then(res => res.blob())
                .then(blob => {
                    this._inputImageObj.src = URL.createObjectURL(blob);
                    this._inputImageObj.onload = () => this.setDirtyCanvas(true, true);
                })
                .catch(e => console.error("Error loading SAM2 input preview:", e));
            }

            try {
                // Draw 2D image preview (200x200)
                ctx.drawImage(this._inputImageObj, 15, 75, 200, 200);
                ctx.strokeStyle = "rgba(255,255,255,0.1)";
                ctx.lineWidth = 1;
                ctx.strokeRect(15, 75, 200, 200);
            } catch (e) {}

            // 3. List detected objects on the right panel
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 13px 'Outfit'";
            ctx.fillText("Đối tượng phát hiện:", 230, 90);

            if (this.properties.crops && this.properties.crops.length > 0) {
                this.properties.crops.forEach((crop, idx) => {
                    const y = 115 + (idx * 45);
                    const colors = ["#fb7185", "#34d399", "#38bdf8", "#facc15"];
                    const color = colors[idx % colors.length];

                    ctx.fillStyle = color;
                    ctx.font = "bold 12px 'Outfit'";
                    ctx.fillText(`✓ [${crop.name}] ${crop.label}`, 230, y);

                    ctx.fillStyle = "#a1a1aa";
                    ctx.font = "10px 'Fira Code'";
                    const conf = crop.confidence ? Math.round(crop.confidence * 100) : 100;
                    const score = crop.mask_score ? Math.round(crop.mask_score * 100) : 100;
                    ctx.fillText(`Conf: ${conf}% | Mask: ${score}%`, 242, y + 16);
                });
            } else {
                ctx.fillStyle = "#52525b";
                ctx.font = "italic 11px 'Outfit'";
                ctx.fillText("(Chờ kết quả GroundingDINO)", 230, 115);
            }

            // 4. Render thumbnails of the crops at the bottom
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 13px 'Outfit'";
            ctx.fillText("Crops tách nền (RGBA):", 15, 305);

            if (this.properties.crops && this.properties.crops.length > 0) {
                if (!this._cropImages || this._cropImages.length === 0) {
                    this._cropImages = [];
                    this.properties.crops.forEach((crop) => {
                        const img = new Image();
                        img._rawUrl = `${backendUrl}${crop.crop_url}`;
                        this._cropImages.push(img);
                        fetch(img._rawUrl, {
                            headers: { "ngrok-skip-browser-warning": "69420" }
                        })
                        .then(res => res.blob())
                        .then(blob => {
                            img.src = URL.createObjectURL(blob);
                            img.onload = () => this.setDirtyCanvas(true, true);
                        })
                        .catch(e => console.error("Error loading crop image:", e));
                    });
                }

                this._cropImages.forEach((img, idx) => {
                    try {
                        const tx = 15 + (idx * 90);
                        // Draw a grid pattern back for alpha visibility
                        ctx.fillStyle = "rgba(255,255,255,0.03)";
                        ctx.fillRect(tx, 320, 80, 80);
                        ctx.strokeStyle = "rgba(255,255,255,0.06)";
                        ctx.strokeRect(tx, 320, 80, 80);

                        ctx.drawImage(img, tx, 320, 80, 80);
                    } catch (e) {}
                });
            } else {
                ctx.fillStyle = "#52525b";
                ctx.font = "italic 11px 'Outfit'";
                ctx.fillText("Chờ xử lý SAM2 segment...", 15, 335);
            }
        };
        LiteGraph.registerNodeType("datn/grounded_sam2", NodeGroundedSAM2);


        // Node 6: TRELLIS 3D Single-Object Reconstruction
        function NodeTrellis3D() {
            this.addInput("crops_data", "object");
            this.addOutput("models_data", "object");
            this.properties = { models: [] };
            this.addWidget("button", "▶ Run to here", () => {
                if (window.runPipelineUntil) window.runPipelineUntil(this);
            });
            this.addWidget("button", "Download GLB models", async () => {
                const files = (this.properties.models || [])
                    .filter(model => model.model_url)
                    .map((model, index) => ({
                        url: `${backendUrl}${model.model_url}`,
                        filename: `${model.name || `object_${index + 1}`}.glb`
                    }));
                if (!files.length) {
                    writeLog("Chưa có model GLB TRELLIS để tải.", "warning");
                    return;
                }
                try {
                    await downloadBackendFiles(files);
                } catch (error) {
                    writeLog(`Lỗi tải model GLB: ${error.message}`, "error");
                }
            });
            this.size = [380, 220];
            this.color = "#1565c0";
            this.bgcolor = "#26282b";
        }
        NodeTrellis3D.title = "🧊 TRELLIS 3D Generator";
        NodeTrellis3D.title_color = "rgba(21, 101, 192, 0.85)";
        NodeTrellis3D.prototype.onExecute = async function () {
            const cropsData = this.getInputData(0);
            if (!cropsData) {
                writeLog("❌ TRELLIS 3D Generator: Thiếu dữ liệu vật thể crops_data!", "error");
                throw new Error("Thiếu dữ liệu vật thể crops_data");
            }
            if (!isConnected) {
                writeLog("❌ TRELLIS 3D Generator: Chưa kết nối GPU Kaggle!", "error");
                throw new Error("Chưa kết nối GPU Kaggle");
            }

            if (this._lastCrops === cropsData && this.properties.models.length > 0) {
                this.setOutputData(0, this.properties.models);
                return;
            }

            this._lastCrops = cropsData;
            writeLog("Chạy khuếch tán 3D TRELLIS Image-to-3D (Mất khoảng 2-3 phút, vui lòng đợi)...", "system");

            try {
                const res = await fetch(`${backendUrl}/api/generate_3d`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "ngrok-skip-browser-warning": "69420"
                    },
                    body: JSON.stringify({ crops: cropsData })
                });
                if (res.ok) {
                    let data = await res.json();

                    // TRELLIS runs as a background job so Ngrok does not have to
                    // keep one request open for several minutes.
                    if (data.job_id) {
                        const jobId = data.job_id;
                        let finished = false;

                        for (let attempt = 0; attempt < 240; attempt++) {
                            await new Promise(resolve => setTimeout(resolve, 5000));

                            const statusRes = await fetch(
                                `${backendUrl}/api/generate_3d/status/${jobId}`,
                                {
                                    headers: {
                                        "ngrok-skip-browser-warning": "69420"
                                    }
                                }
                            );

                            if (!statusRes.ok) {
                                const statusError = await statusRes.text();
                                throw new Error(
                                    `TRELLIS status error: ${statusRes.status} - ${statusError}`
                                );
                            }

                            data = await statusRes.json();

                            if (data.status === "completed") {
                                finished = true;
                                break;
                            }

                            if (data.status === "failed") {
                                throw new Error(data.error || "TRELLIS job failed");
                            }

                            if (attempt % 6 === 0) {
                                if (data.status === "queued") {
                                    writeLog(
                                        `TRELLIS đang xếp hàng GPU... vị trí ${data.queue_position || "?"}`,
                                        "system"
                                    );
                                } else {
                                    writeLog(
                                        `TRELLIS đang xử lý trên Kaggle... (${Math.round((attempt + 1) * 5 / 60)} phút)`,
                                        "system"
                                    );
                                }
                            }
                        }

                        if (!finished) {
                            throw new Error("TRELLIS quá thời gian chờ 20 phút");
                        }
                    }

                    this.properties.models = data.models || [];
                    this.setOutputData(0, this.properties.models);
                    writeLog("✅ TRELLIS dựng hình 3D Mesh hoàn thành!", "success");
                    this.setDirtyCanvas(true, true);
                } else {
                    const errText = await res.text();
                    throw new Error(`API error: ${res.status} - ${errText}`);
                }
            } catch (e) {
                writeLog(`❌ Lỗi chạy TRELLIS: ${e.message}`, "error");
                throw e;
            }
        };
        NodeTrellis3D.prototype.onDrawForeground = function (ctx) {
            if (!this.properties.models || this.properties.models.length === 0) {
                ctx.fillStyle = "#52525b";
                ctx.font = "italic 13px 'Outfit'";
                ctx.fillText("Chờ xử lý mô hình 3D...", 20, 60);
                return;
            }

            ctx.fillStyle = "#818cf8";
            ctx.font = "bold 13px 'Outfit'";
            ctx.fillText("Mô hình GLB trích xuất:", 15, 60);

            this.properties.models.forEach((m, idx) => {
                ctx.fillStyle = "#34d399";
                ctx.font = "12px 'Outfit'";
                ctx.fillText(`✓ [${m.name}] ${m.label}:`, 20, 85 + (idx * 34));

                ctx.fillStyle = "#71717a";
                ctx.font = "10px 'Fira Code'";
                ctx.fillText(m.model_url.substring(0, 52) + "...", 25, 99 + (idx * 34));
            });
        };
        LiteGraph.registerNodeType("datn/trellis_3d", NodeTrellis3D);


        // Node 7: 3D Scene Combiner (Integrates the Interactive 3D Model-Viewer Web Component)
        function NodeSceneCombiner() {
            this.addInput("models_data", "object");
            this.addOutput("scene_url", "string");

            this.properties = {
                sceneUrl: "",
                scaleFactor: 0.01
            };
            this.addWidget("button", "▶ Run to here", () => {
                if (window.runPipelineUntil) window.runPipelineUntil(this);
            });

            this.addWidget("number", "3D System Scale", this.properties.scaleFactor, (v) => {
                this.properties.scaleFactor = v;
            }, { min: 0.002, max: 0.05, step: 0.002 });

            this.size = [420, 400];
            this.color = "#6a1b9a";
            this.bgcolor = "#26282b";
        }
        NodeSceneCombiner.title = "🌐 3D Scene Combiner";
        NodeSceneCombiner.title_color = "rgba(106, 27, 154, 0.85)";
        NodeSceneCombiner.prototype.onExecute = async function () {
            const modelsData = this.getInputData(0);
            if (!modelsData) {
                writeLog("❌ 3D Scene Combiner: Thiếu dữ liệu mô hình models_data!", "error");
                throw new Error("Thiếu dữ liệu mô hình models_data");
            }
            if (!isConnected) {
                writeLog("❌ 3D Scene Combiner: Chưa kết nối GPU Kaggle!", "error");
                throw new Error("Chưa kết nối GPU Kaggle");
            }

            if (this._lastModels === modelsData && this.properties.sceneUrl) {
                this.setOutputData(0, this.properties.sceneUrl);
                return;
            }

            this._lastModels = modelsData;
            writeLog("Trimesh đang căn chỉnh tọa độ và đóng gói file Scene 3D...", "system");

            try {
                const res = await fetch(`${backendUrl}/api/combine_scene`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "ngrok-skip-browser-warning": "69420"
                    },
                    body: JSON.stringify({
                        models: modelsData,
                        layout: {}, // Send empty layout to satisfy backend schema
                        scale_factor: this.properties.scaleFactor
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    this.properties.sceneUrl = `${backendUrl}${data.scene_url}`;
                    this.properties.zipUrl = `${backendUrl}${data.zip_url}`;
                    this.setOutputData(0, this.properties.sceneUrl);
                    writeLog("🎉 HỆ THỐNG ĐÃ HOÀN TẤT DỰNG CẢNH 3D!", "success");

                    this.update3DViewer();
                } else {
                    const errText = await res.text();
                    throw new Error(`API error: ${res.status} - ${errText}`);
                }
            } catch (e) {
                writeLog(`❌ Lỗi ghép cảnh: ${e.message}`, "error");
                throw e;
            }
        };

        // Update overlay interactive 3D view on DOM
        NodeSceneCombiner.prototype.update3DViewer = function () {
            if (!this.properties.sceneUrl) return;

            const viewerId = `viewer-${this.id}`;
            let viewer = document.getElementById(viewerId);

            if (!viewer) {
                viewer = document.createElement("div");
                viewer.id = viewerId;
                viewer.className = "node-3d-container";
                viewer.style.border = "1px solid rgba(255, 255, 255, 0.1)";
                viewer.style.borderRadius = "8px";
                viewer.style.zIndex = 1000;

                document.getElementById("canvas-container").appendChild(viewer);
            }

            if (this._lastGLBUrl !== this.properties.sceneUrl) {
                this._lastGLBUrl = this.properties.sceneUrl;
                
                viewer.innerHTML = `<div style="color: #a1a1aa; font-family: 'Outfit'; display: flex; align-items: center; justify-content: center; height: 100%; background: #0b0b0e;">⏳ Đang nạp mô hình 3D...</div>`;
                
                fetch(this.properties.sceneUrl, {
                    headers: { "ngrok-skip-browser-warning": "69420" }
                })
                .then(res => {
                    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                    return res.blob();
                })
                .then(blob => {
                    if (this._objectUrl) {
                        URL.revokeObjectURL(this._objectUrl);
                    }
                    this._objectUrl = URL.createObjectURL(blob);
                    syncSpatialViewport(this._objectUrl, this.properties.sceneUrl, this.properties.zipUrl);
                    
                    viewer.innerHTML = `
                    <div class="viewer-stage">
                        <model-viewer 
                        src="${this._objectUrl}" 
                        camera-controls 
                        auto-rotate 
                        alt="DATN Interactive 3D Scene Combined Output"
                        shadow-intensity="1.5"
                        exposure="1.0"
                        style="background-color: #0b0b0e; width: 100%; height: 100%;">
                        </model-viewer>
                        <div class="viewer-controls">
                            <button class="viewer-btn" id="btn-spin-${this.id}">Stop Spin</button>
                            <button class="viewer-btn" id="btn-reset-${this.id}">Reset View</button>
                            <button class="viewer-btn" id="btn-scene-${this.id}">Tải GLB</button>
                            <button class="viewer-btn" id="btn-download-${this.id}" style="background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%); color: white; border-color: rgba(255,255,255,0.2);">📦 Tải gói ZIP</button>
                        </div>
                    </div>
                    `;

                    const modelViewer = viewer.querySelector("model-viewer");

                    viewer.querySelector(`#btn-spin-${this.id}`).onclick = (e) => {
                        e.stopPropagation();
                        if (modelViewer.hasAttribute("auto-rotate")) {
                            modelViewer.removeAttribute("auto-rotate");
                            e.currentTarget.innerText = "Auto Spin";
                        } else {
                            modelViewer.setAttribute("auto-rotate", "");
                            e.currentTarget.innerText = "Stop Spin";
                        }
                    };

                    viewer.querySelector(`#btn-reset-${this.id}`).onclick = (e) => {
                        e.stopPropagation();
                        modelViewer.cameraOrbit = "0deg 75deg auto";
                        modelViewer.fieldOfView = "auto";
                    };

                    viewer.querySelector(`#btn-scene-${this.id}`).onclick = async (e) => {
                        e.stopPropagation();
                        try {
                            await downloadBackendFile(
                                this.properties.sceneUrl,
                                "scene_combined.glb"
                            );
                        } catch (error) {
                            writeLog(`Lỗi tải scene GLB: ${error.message}`, "error");
                        }
                    };

                    viewer.querySelector(`#btn-download-${this.id}`).onclick = async (e) => {
                        e.stopPropagation();
                        if (this.properties.zipUrl) {
                            const button = e.currentTarget;
                            const originalText = button.innerText;
                            button.disabled = true;
                            button.innerText = "Đang tải...";

                            try {
                                const downloadUrl = new URL(this.properties.zipUrl);
                                downloadUrl.searchParams.set("ngrok-skip-browser-warning", "1");

                                const response = await fetch(downloadUrl.toString(), {
                                    headers: {
                                        "ngrok-skip-browser-warning": "1"
                                    }
                                });

                                if (!response.ok) {
                                    throw new Error(`HTTP ${response.status}`);
                                }

                                const blob = await response.blob();
                                const objectUrl = URL.createObjectURL(blob);
                                const a = document.createElement("a");
                                a.href = objectUrl;
                                a.download = "scene_assets.zip";
                                document.body.appendChild(a);
                                a.click();
                                a.remove();
                                URL.revokeObjectURL(objectUrl);
                                writeLog("📦 Đã tải gói ZIP tài nguyên xuống máy.", "success");
                            } catch (error) {
                                writeLog(`❌ Không thể tải gói ZIP: ${error.message}`, "error");
                            } finally {
                                button.disabled = false;
                                button.innerText = originalText;
                            }
                        } else {
                            writeLog("❌ Lỗi: Link tải ZIP chưa sẵn sàng.", "error");
                        }
                    };
                })
                .catch(err => {
                    writeLog(`❌ Lỗi nạp mô hình 3D: ${err.message}`, "error");
                    setSpatialViewportError("Preview failed");
                    viewer.innerHTML = `<div style="color: #f44336; font-family: 'Outfit'; display: flex; align-items: center; justify-content: center; height: 100%; background: #0b0b0e; padding: 10px; text-align: center;">❌ Lỗi nạp mô hình 3D: ${err.message}</div>`;
                });
            }
            this.position3DViewer();
        };

        NodeSceneCombiner.prototype.position3DViewer = function () {
            const viewerId = `viewer-${this.id}`;
            const viewer = document.getElementById(viewerId);
            if (!viewer) return;

            const scale = canvas.ds.scale;
            const nodeWidth = this.size[0];
            const nodeHeight = this.size[1];

            // Calculate screen bounding boxes mapping LiteGraph coords to DOM layout
            const screenPos = canvas.convertOffsetToCanvas([this.pos[0] + 15, this.pos[1] + 95]);
            const x = screenPos[0];
            const y = screenPos[1];
            const w = (nodeWidth - 30) * scale;
            const h = (nodeHeight - 115) * scale;

            viewer.style.left = `${x}px`;
            viewer.style.top = `${y}px`;
            viewer.style.width = `${w}px`;
            viewer.style.height = `${h}px`;

            // Prevent rendering when zoomed out excessively to optimize drawing performance
            if (scale < 0.25) {
                viewer.style.display = "none";
            } else {
                viewer.style.display = "block";
            }
        };

        // Auto align overlays when zooming or dragging canvas
        const originalDraw = LGraphCanvas.prototype.draw;
        LGraphCanvas.prototype.draw = function () {
            originalDraw.apply(this, arguments);

            const zoomLabel = document.getElementById("canvas-zoom-label");
            if (zoomLabel) {
                zoomLabel.innerText = `Zoom ${Math.round(canvas.ds.scale * 100)}%`;
            }

            // Căn chỉnh vị trí 3D Viewer
            const combiners = graph.findNodesByType("datn/scene_combiner");
            if (combiners && combiners.length > 0) {
                combiners.forEach(n => n.position3DViewer());
            }

            // Căn chỉnh vị trí Textarea nhập Prompt
            const prompts = graph.findNodesByType("datn/input_prompt");
            if (prompts && prompts.length > 0) {
                prompts.forEach(n => n.updateTextarea());
            }
        };

        NodeSceneCombiner.prototype.onRemoved = function () {
            const viewerId = `viewer-${this.id}`;
            const viewer = document.getElementById(viewerId);
            if (viewer) viewer.remove();
        };

        LiteGraph.registerNodeType("datn/scene_combiner", NodeSceneCombiner);

        // =========================================================================
        // 3. GRAPH SETUP & DEFAULT TEMPLATE
        // =========================================================================
        function setupDefaultGraph() {
            graph.clear();

            // Clean linear pipeline layout:
            // [Prompt] ──▶ [Gemini Optimizer] ──▶ [SD3.5] ──▶ [SAM2] ──▶ [TRELLIS] ──▶ [Combiner]

            const nodePrompt = LiteGraph.createNode("datn/input_prompt");
            nodePrompt.pos = [100, 200];
            graph.add(nodePrompt);

            const nodeOptimizer = LiteGraph.createNode("datn/prompt_optimizer");
            nodeOptimizer.pos = [500, 200];
            graph.add(nodeOptimizer);

            const nodeSDXL = LiteGraph.createNode("datn/sdxl_gen");
            nodeSDXL.pos = [950, 200];
            graph.add(nodeSDXL);

            const nodeSAM2 = LiteGraph.createNode("datn/grounded_sam2");
            nodeSAM2.pos = [1480, 200];
            graph.add(nodeSAM2);

            const nodeTrellis = LiteGraph.createNode("datn/trellis_3d");
            nodeTrellis.pos = [1960, 200];
            graph.add(nodeTrellis);

            const nodeCombiner = LiteGraph.createNode("datn/scene_combiner");
            nodeCombiner.pos = [2380, 200];
            graph.add(nodeCombiner);

            // Connect Node slots directly
            nodePrompt.connect(0, nodeOptimizer, 0);
            nodeOptimizer.connect(0, nodeSDXL, 0);
            nodeSDXL.connect(0, nodeSAM2, 0);
            nodePrompt.connect(0, nodeSAM2, 2);
            nodeSAM2.connect(0, nodeTrellis, 0);
            nodeTrellis.connect(0, nodeCombiner, 0);

            // Fit everything on screen
            canvas.ds.offset = [80, 50];
            canvas.ds.scale = 0.52;

            writeLog("Sơ đồ luồng pipeline mặc định đã được nạp thành công.", "info");
        }

        let isExecuting = false;
        document.getElementById("btn-run").onclick = async () => {
            if (isExecuting) return;
            if (!isConnected) {
                writeLog("Vui lòng kết nối tới server GPU Kaggle bằng Ngrok trước!", "warning");
                return;
            }
            
            isExecuting = true;
            const btnRun = document.getElementById("btn-run");
            const originalHTML = btnRun.innerHTML;
            const queueState = document.getElementById("queue-state");
            const queueStateLabel = document.getElementById("queue-state-label");
            const queueCount = document.getElementById("queue-count");

            queueState.className = "queue-state running";
            queueStateLabel.innerText = "Running workflow...";
            queueCount.innerText = "1 running";
            btnRun.disabled = true;
            btnRun.style.opacity = "0.6";
            btnRun.style.cursor = "not-allowed";
            btnRun.innerHTML = `<span>⏳ Đang xử lý...</span>`;
            
            writeLog("🏁 Bắt đầu thực thi toàn bộ quy trình (Queue Prompt)...", "system");
            
            const promptNode = graph.findNodesByType("datn/input_prompt")[0];
            const sdxlNode = graph.findNodesByType("datn/sdxl_gen")[0];
            const sam2Node = graph.findNodesByType("datn/grounded_sam2")[0];
            const trellisNode = graph.findNodesByType("datn/trellis_3d")[0];
            const combinerNode = graph.findNodesByType("datn/scene_combiner")[0];
 
            const runNode = async (node, nodeName) => {
                if (!node) return;
                node.boxcolor = "#00ff00"; // Highlight executing node with green border
                canvas.draw(true, true);
                writeLog(`⏳ Đang thực thi: ${nodeName}...`, "system");
                try {
                    await node.onExecute();
                    node.boxcolor = "rgba(255, 255, 255, 0.1)"; // Restore border
                    canvas.draw(true, true);
                } catch (err) {
                    node.boxcolor = "#f44336"; // Highlight error node with red border
                    canvas.draw(true, true);
                    throw err;
                }
            };
 
            try {
                if (promptNode) {
                    await runNode(promptNode, "Input Text Prompt");
                }
                if (sdxlNode) {
                    await runNode(sdxlNode, "SD3.5 Image Generator");
                }
                if (sam2Node) {
                    await runNode(sam2Node, "Grounded-SAM2 Segment");
                }
                if (trellisNode) {
                    await runNode(trellisNode, "TRELLIS 3D Generator");
                }
                if (combinerNode) {
                    await runNode(combinerNode, "3D Scene Combiner");
                }
                writeLog("🎉 Thực thi toàn bộ quy trình pipeline THÀNH CÔNG!", "success");
                queueState.className = "queue-state success";
                queueStateLabel.innerText = "Completed successfully";
            } catch (err) {
                writeLog("❌ Quy trình dừng lại do lỗi: " + err.message, "error");
                queueState.className = "queue-state error";
                queueStateLabel.innerText = "Execution failed";
            } finally {
                isExecuting = false;
                queueCount.innerText = "0 running";
                btnRun.disabled = false;
                btnRun.style.opacity = "1";
                btnRun.style.cursor = "pointer";
                btnRun.innerHTML = originalHTML;
            }
        };

        async function runPipelineUntil(targetNode) {
            if (isExecuting) return;
            if (!isConnected) {
                writeLog("Vui lòng kết nối tới server GPU Kaggle trước.", "warning");
                return;
            }

            const nodes = [
                [graph.findNodesByType("datn/input_prompt")[0], "Input Text Prompt"],
                [graph.findNodesByType("datn/prompt_optimizer")[0], "Gemini Prompt Optimizer"],
                [graph.findNodesByType("datn/sdxl_gen")[0], "SD3.5 Image Generator"],
                [graph.findNodesByType("datn/grounded_sam2")[0], "Grounded-SAM2 Segment"],
                [graph.findNodesByType("datn/trellis_3d")[0], "TRELLIS 3D Generator"],
                [graph.findNodesByType("datn/scene_combiner")[0], "3D Scene Combiner"]
            ];
            const endIndex = nodes.findIndex(([node]) => node === targetNode);
            if (endIndex < 0) return;

            isExecuting = true;
            const btnRun = document.getElementById("btn-run");
            const queueState = document.getElementById("queue-state");
            const queueStateLabel = document.getElementById("queue-state-label");
            const queueCount = document.getElementById("queue-count");
            const originalHTML = btnRun.innerHTML;

            queueState.className = "queue-state running";
            queueStateLabel.innerText = "Running to selected node...";
            queueCount.innerText = "1 running";
            btnRun.disabled = true;
            btnRun.innerHTML = "⏳ Running...";

            try {
                for (let i = 0; i <= endIndex; i++) {
                    const [node, name] = nodes[i];
                    if (!node) continue;
                    node.boxcolor = "#00ff00";
                    canvas.draw(true, true);
                    writeLog(`⏳ Đang thực thi: ${name}...`, "system");
                    await node.onExecute();
                    node.boxcolor = "rgba(255, 255, 255, 0.1)";
                    canvas.draw(true, true);
                }

                queueState.className = "queue-state success";
                queueStateLabel.innerText = `Đã dừng tại: ${nodes[endIndex][1]}`;
                writeLog(`✅ Hoàn tất đến node: ${nodes[endIndex][1]}.`, "success");
            } catch (error) {
                queueState.className = "queue-state error";
                queueStateLabel.innerText = "Execution failed";
                writeLog(`❌ Quy trình dừng lại do lỗi: ${error.message}`, "error");
            } finally {
                isExecuting = false;
                queueCount.innerText = "0 running";
                btnRun.disabled = false;
                btnRun.innerHTML = originalHTML;
                canvas.draw(true, true);
            }
        }

        window.runPipelineUntil = runPipelineUntil;

        document.getElementById("btn-run").onclick = () => {
            runPipelineUntil(graph.findNodesByType("datn/scene_combiner")[0]);
        };

        setupDefaultGraph();

        // ComfyUI-style node palette and canvas controls
        const nodeSearch = document.getElementById("node-search");
        const paletteItems = Array.from(
            document.querySelectorAll(".node-palette-item")
        );

        nodeSearch.addEventListener("input", () => {
            const query = nodeSearch.value.trim().toLowerCase();
            paletteItems.forEach((item) => {
                item.style.display = item.textContent
                    .toLowerCase()
                    .includes(query)
                    ? "flex"
                    : "none";
            });
        });

        paletteItems.forEach((item) => {
            item.addEventListener("click", () => {
                const node = LiteGraph.createNode(item.dataset.nodeType);
                if (!node) return;

                const visible = canvas.visible_area;
                node.pos = [
                    visible[0] + visible[2] * 0.5 - node.size[0] * 0.5,
                    visible[1] + visible[3] * 0.5 - node.size[1] * 0.5
                ];

                graph.add(node);
                canvas.selectNode(node);
                canvas.draw(true, true);
                writeLog(`Đã thêm node: ${item.textContent.trim()}`, "info");
            });
        });

        document.getElementById("btn-fit-view").onclick = () => {
            canvas.ds.offset = [80, 50];
            canvas.ds.scale = 0.52;
            canvas.draw(true, true);
        };

        document.getElementById("btn-reset-workflow").onclick = () => {
            document
                .querySelectorAll(".node-textarea, .node-3d-container")
                .forEach((element) => element.remove());
            setupDefaultGraph();
            canvas.draw(true, true);
        };

        window.addEventListener("keydown", (event) => {
            if (event.ctrlKey && event.key === "Enter") {
                event.preventDefault();
                document.getElementById("btn-run").click();
            }
        });


        // Kaggle Modal Events
        document.addEventListener("DOMContentLoaded", () => {
            const modal = document.getElementById("kaggle-modal");
            const btnHelp = document.getElementById("btn-kaggle-help");
            const btnClose = document.getElementById("close-modal");
            const btnCopy = document.getElementById("btn-copy-compile");

            if (btnHelp) btnHelp.onclick = () => { modal.style.display = "flex"; };
            if (btnClose) btnClose.onclick = () => { modal.style.display = "none"; };
            window.onclick = (e) => { if (e.target === modal) modal.style.display = "none"; };
            if (btnCopy) btnCopy.onclick = () => {
                const codeText = document.getElementById("code-compile").innerText;
                navigator.clipboard.writeText(codeText).then(() => {
                    btnCopy.innerText = "Copied!";
                    setTimeout(() => { btnCopy.innerText = "Copy Code"; }, 2000);
                }).catch(err => console.error("Copy error:", err));
            };
        });
    