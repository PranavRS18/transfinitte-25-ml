import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import 'dotenv/config';

const app = express();
const PORT = process.env.PORT || 5000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
app.use(express.json());
// app.use(bodyParser.json()); Alternative if Needed

// Routes
app.get('/', (req, res) => {
  res.send("Hello! Welcome to the TransfiNITTe'25 API Server");
});

app.post('/api/summarize', (req, res) => {
  let { text, model_name } = req.body;
  if (!text) return res.status(400).json({ error: 'Text is required' });

  if (!model_name) model_name = 'llama3.2:1b';

  // Set Provider based on model_name
  let provider = 'ollama';
  if (model_name && model_name.toLowerCase().includes('gemini')) {
    provider = 'gemini';
  }
  console.log(`Provider: ${provider}, Model: ${model_name}, Text: ${text}`);

  const pythonScript = path.join(__dirname, './utils/model/summarizer.py');

  // Spawn Python process
  const pyProcess = spawn('python', [pythonScript, provider, model_name, text]);

  let result = '';
  let error = '';
  
  pyProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pyProcess.stderr.on('data', (data) => {
    error += data.toString();
  });

  pyProcess.on('close', (code) => {
    if (code !== 0 || error) {
      console.error('Python error:', error);
      return res.status(500).json({ error: 'Python script failed', details: error });
    }
    console.log("Result: ", result);
    res.json({ summary: result.trim() });
  });
});

app.post('/api/chatbot', (req, res) => {
  const { text, model_name } = req.body;
  if (!text) return res.status(400).json({ error: 'Text is required' });

  // Automatically set provider based on model_name
  let provider = 'ollama';
  if (model_name && model_name.toLowerCase().includes('gemini')) {
    provider = 'gemini';
  }

  const pythonScript = path.join(__dirname, './utils/model/chatbot.py');

  // Spawn Python process
  const pyProcess = spawn('python', [pythonScript, provider, model_name || 'llama3.2:1b', text]);

  let result = '';
  let error = '';

  pyProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pyProcess.stderr.on('data', (data) => {
    error += data.toString();
  });

  pyProcess.on('close', (code) => {
    if (code !== 0 || error) {
      console.error('Python error:', error);
      return res.status(500).json({ error: 'Python script failed', details: error });
    }

    res.json({reply: result.trim() });
  });
});
// Route to Add Documents to VectorDB
app.post('/api/add-docs', async (req, res) => {
  const { userId, text } = req.body;
  if (!userId || !text) {
    return res.status(400).json({ error: 'userId and text required' });
  }
  console.log("text",text);
  try {
    const pythonScript = path.join(__dirname, './utils/model/rag_user.py');

    // Spawn Python process
    const pyProcess = spawn('python', [pythonScript, 'add', userId, text]);

    let output = '';
    let error = '';

    pyProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pyProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pyProcess.on('close', (code) => {
      if (code !== 0 || error) {
        console.error('Python error:', error);
        return res.status(500).json({ error: error || 'Python script failed' });
      }
      try {
        res.json({ success: output ? true : false });
      } catch (parseErr) {
        res.status(500).json({ error: 'Failed to parse Python output' });
      }
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Route to Query RAG
app.post('/api/query', async (req, res) => {
  const { userId, query, model_name } = req.body;
  if (!userId || !query || !model_name)
    return res.status(400).json({ error: 'userId, query, and model_name required' });

  try {
    const pythonScript = path.join(__dirname, './utils/model/rag_user.py');

    const py = spawn('python', [
      pythonScript,
      'query',
      userId,
      query,
      model_name, // send as model_name to match Python Script
    ]);

    let output = '';
    let error = '';

    py.stdout.on('data', (data) => {
      output += data.toString();
    });
    py.stderr.on('data', (data) => {
      error += data.toString();
    });

    py.on('close', (code) => {
      if (code !== 0 || error) {
        console.error('Python error:', error);
        return res.status(500).json({ error: error || 'Python script failed' });
      }
      try {
        res.json(JSON.parse(output));
      } catch (parseErr) {
        res.status(500).json({ error: 'Failed to parse Python output' });
      }
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on Port ${PORT}`);
});
