const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const app = express();

const PORT = 5000;

// Middleware
app.use(express.json()); // parse JSON requests
// app.use(bodyParser.json()); // alternative if needed

// Routes
app.get("/", (req, res) => {
  res.send("Hello! Server is running.");
});
app.post("/api/summarize", (req, res) => {
  const { text, model_name } = req.body;
  if (!text) return res.status(400).json({ error: "Text is required" });

  // Automatically set provider based on model_name
  let provider = "ollama";
  if (model_name && model_name.toLowerCase().includes("gemini")) {
    provider = "gemini";
  }
  console.log(provider)

  const pythonScript = path.join(__dirname, "../utils/model/summarise.py");

  // Spawn Python process
  const pyProcess = spawn("python", [pythonScript, provider, model_name || "llama3.2:1b", text]);

  let result = "";
  let error = "";

  pyProcess.stdout.on("data", (data) => {
    result += data.toString();
  });

  pyProcess.stderr.on("data", (data) => {
    error += data.toString();
  });

  pyProcess.on("close", (code) => {
    if (code !== 0 || error) {
      console.error("Python error:", error);
      return res.status(500).json({ error: "Python script failed", details: error });
    }

    res.json({ summary: result.trim() });
  });
});



app.post("/api/add-docs", async (req, res) => {
  const { user_id, documents } = req.body;
  if (!user_id || !documents) return res.status(400).json({ error: "user_id and documents required" });

  try {
    // Call Python script via child_process
    const py = spawn("python", ["-c", `
import sys, json
sys.path.append("..")  # go to project root
from utils.model.user_rag import add_documents
result = add_documents("${user_id}", ${JSON.stringify(documents)})
print(json.dumps(result))
    `]);

    let output = "";
    py.stdout.on("data", (data) => { output += data.toString(); });
    py.stderr.on("data", (data) => { console.error(data.toString()); });

    py.on("close", () => {
      res.json(JSON.parse(output));
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Route to query RAG
app.post("/api/query", async (req, res) => {
  const { user_id, query } = req.body;
  if (!user_id || !query) return res.status(400).json({ error: "user_id and query required" });

  try {
    const py = spawn("python", ["-c", `
import sys, json
sys.path.append("..")
from utils.model.user_rag import rag_query
result = rag_query("${user_id}", "${query}")
print(json.dumps(result))
    `]);

    let output = "";
    py.stdout.on("data", (data) => { output += data.toString(); });
    py.stderr.on("data", (data) => { console.error(data.toString()); });

    py.on("close", () => {
      res.json(JSON.parse(output));
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});


