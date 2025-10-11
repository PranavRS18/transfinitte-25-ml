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



// Route to add documents for a user
app.post("/api/add-docs", async (req, res) => {
  const { user_id, documents } = req.body;
  if (!user_id || !documents || !Array.isArray(documents)) {
    return res.status(400).json({ error: "user_id and documents (array) required" });
  }

  try {
    const pythonScript = path.join(__dirname, "../utils/model/user_rag.py");

    const py = spawn("python", [pythonScript, "add", user_id, JSON.stringify(documents)]);

    let output = "";
    let error = "";

    py.stdout.on("data", (data) => { output += data.toString(); });
    py.stderr.on("data", (data) => { error += data.toString(); });

    py.on("close", (code) => {
      if (code !== 0 || error) {
        console.error("Python error:", error);
        return res.status(500).json({ error: error || "Python script failed" });
      }
      try {
        res.json(JSON.parse(output));
      } catch (parseErr) {
        res.status(500).json({ error: "Failed to parse Python output" });
      }
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Route to query RAG
app.post("/api/query", async (req, res) => {
  const { user_id, query, model_name } = req.body;
  if (!user_id || !query || !model_name)
    return res.status(400).json({ error: "user_id, query, and model_name required" });

  try {
    const pythonScript = path.join(__dirname, "../utils/model/user_rag.py");

    const py = spawn("python", [
      pythonScript,
      "query",
      user_id,
      query,
      model_name // send as model_name to match Python
    ]);

    let output = "";
    let error = "";

    py.stdout.on("data", (data) => { output += data.toString(); });
    py.stderr.on("data", (data) => { error += data.toString(); });

    py.on("close", (code) => {
      if (code !== 0 || error) {
        console.error("Python error:", error);
        return res.status(500).json({ error: error || "Python script failed" });
      }
      try {
        res.json(JSON.parse(output));
      } catch (parseErr) {
        res.status(500).json({ error: "Failed to parse Python output" });
      }
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on Port ${PORT}`);
});


