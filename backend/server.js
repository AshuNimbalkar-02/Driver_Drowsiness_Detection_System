import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, 'data');
const LOGS_FILE = path.join(DATA_DIR, 'logs.json');

// Ensure data folder exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Ensure logs.json exists
if (!fs.existsSync(LOGS_FILE)) {
  fs.writeFileSync(LOGS_FILE, JSON.stringify([]));
}

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

// Get all logs
app.get('/api/logs', (req, res) => {
  try {
    const data = fs.readFileSync(LOGS_FILE, 'utf8');
    res.json(JSON.parse(data));
  } catch (error) {
    res.status(500).json({ error: 'Failed to read logs database.' });
  }
});

// Get logs by Fleet ID
app.get('/api/logs/:fleetId', (req, res) => {
  const { fleetId } = req.params;
  try {
    const data = fs.readFileSync(LOGS_FILE, 'utf8');
    const logs = JSON.parse(data);
    const filtered = logs.filter(log => log.fleetId === fleetId);
    res.json(filtered);
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve logs for fleet.' });
  }
});

// Post a new safety log event
app.post('/api/logs', (req, res) => {
  const { msg, type, safetyScore, ear, mar, fleetId } = req.body;

  if (!msg) {
    return res.status(400).json({ error: 'Log message is required.' });
  }

  const newLog = {
    id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
    time: new Date().toLocaleTimeString(),
    timestamp: new Date().toISOString(),
    msg,
    type: type || 'info',
    safetyScore: safetyScore ?? 100,
    ear: ear || 0,
    mar: mar || 0,
    fleetId: fleetId || 'UNKNOWN'
  };

  try {
    const data = fs.readFileSync(LOGS_FILE, 'utf8');
    const logs = JSON.parse(data);
    logs.unshift(newLog);
    // Keep last 200 logs to prevent file bloat
    const trimmedLogs = logs.slice(0, 200);
    fs.writeFileSync(LOGS_FILE, JSON.stringify(trimmedLogs, null, 2));
    res.status(201).json(newLog);
  } catch (error) {
    res.status(500).json({ error: 'Failed to write log to database.' });
  }
});

// Sync endpoint simulating fleet registration / config sync
app.post('/api/sync', (req, res) => {
  const { fleetId, stats } = req.body;
  console.log(`Syncing parameters for Fleet ID: ${fleetId}`);
  
  res.json({
    status: 'success',
    timestamp: new Date().toISOString(),
    message: `Fleet ${fleetId} successfully synced with DrivenGuard Cloud Central.`,
    nodeStatus: 'ACTIVE',
    serverLatency: '12.4ms',
    databaseIntegrity: '100%'
  });
});

app.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`🚀 DrivenGuard Edge Backend running on port ${PORT}`);
  console.log(`📂 DB Location: ${LOGS_FILE}`);
  console.log(`====================================================`);
});
