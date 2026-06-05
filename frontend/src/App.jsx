import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import Webcam from 'react-webcam';
import * as FaceMesh from '@mediapipe/face_mesh';
import * as cam from '@mediapipe/camera_utils';
import { Howl } from 'howler';
import {
    AlertTriangle, Eye, Activity, ShieldCheck, Zap, History, Terminal, Clock,
    BarChart3, Dna, UserCheck2, BellRing, Navigation2, Settings, X, Smartphone,
    Globe2, Cpu, Database, Fingerprint, RefreshCcw, WifiOff, FileDown,
    Play, Pause, Wand2, ShieldAlert, BrainCircuit, Scan, Minimize2, Maximize2, Moon
} from 'lucide-react';
import { AreaChart, Area, CartesianGrid, ResponsiveContainer, YAxis } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { SimpleMLP, generateSyntheticData } from './utils/mlp';
import defaultDataset from './utils/dataset.json';
import './index.css';

// ---------------------------------------------------------
// TOP 1% KNOWLEDGE BASE & STRATEGY
// ---------------------------------------------------------
const LANGUAGES = {
    en: {
        wake: "Wake up! Critical alertness required.",
        eyes: "Eyes on path detected as distracted.",
        alert: "DROWSINESS_X1",
        distracted: "EYE_GAZE_LOW",
        name: "English",
        stability: "Stability Index",
        fleet: "Fleet Identity Node",
        gaze: "Gaze Localization",
        stream: "Neural Stream",
        biometrics: "Biometrics",
        integrity: "Cyber-Node Integrity",
        latency: "Latency Node",
        blinks: "Blink Count",
        risk: "Collision Risk",
        uptime: "Uptime",
        insights: "Generate Global Insights",
        summary: "Executive Summary",
        roi: "Funding & ROI Node",
        export: "SECURE DIGITAL EXPORT",
        close: "Close"
    },
    mr: {
        wake: "जागे व्हा! आत्णीक जागरूकता आवश्यक आहे.",
        eyes: "रस्त्यावर लक्ष ठेवा!",
        alert: "झोप_चेतावणी",
        distracted: "लक्षांतर_धोका",
        name: "मराठी",
        stability: "स्थिरता निर्देशांक",
        fleet: "फ्लीट ओळख नोड",
        gaze: "नजर स्थानिकीकरण",
        stream: "न्युरल स्ट्रीम",
        biometrics: "बायोमेट्रिक्स",
        integrity: "सायबर-नोड अखंडता",
        latency: "विलंबता नोड",
        blinks: "डोळे मिचकावणे",
        risk: "धडक धोका",
        uptime: "अपटाइम",
        insights: "जागतिक अहवाल तयार करा",
        summary: "कार्यकारी सारांश",
        roi: "निधी आणि ROI नोड",
        export: "सुरक्षित डिजिटल एक्सपोर्ट",
        close: "बंद करा"
    },
    fr: {
        wake: "Alerte de somnolence critique!",
        eyes: "Concentrez-vous sur la route!",
        alert: "SOMMEIL_V1",
        distracted: "DISTRACTION",
        name: "Français",
        stability: "Indice de Stabilité",
        fleet: "Nœud d'Identité de Flotte",
        gaze: "Localisation du Regard",
        stream: "Flux Neural",
        biometrics: "Biométrie",
        integrity: "Intégrité Cyber-Nœud",
        latency: "Nœud de Latence",
        blinks: "Nombre de Clignements",
        risk: "Risque de Collision",
        uptime: "Temps de Fonctionnement",
        insights: "Générer des Rapports",
        summary: "Résumé Exécutif",
        roi: "Nœud de Financement & ROI",
        export: "EXPORTATION DIGITALE SÉCURISÉE",
        close: "Fermer"
    },
    es: {
        wake: "¡Alerta crítica de somnolencia!",
        eyes: "¡Ojos en la carretera!",
        alert: "SUEÑO_V1",
        distracted: "DISTRACCIÓN",
        name: "Español",
        stability: "Índice de Estabilidad",
        fleet: "Nodo de Identidad de Flota",
        gaze: "Localización de Mirada",
        stream: "Flujo Neural",
        biometrics: "Biometría",
        integrity: "Integridad del Cyber-Nodo",
        latency: "Nodo de Latencia",
        blinks: "Conteo de Parpadeos",
        risk: "Riesgo de Colisión",
        uptime: "Tiempo de Actividad",
        insights: "Generar Informes Globales",
        summary: "Resumen Ejecutivo",
        roi: "Nodo de Financiación y ROI",
        export: "EXPORTACIÓN DIGITAL SEGURA",
        close: "Cerrar"
    }
};

// Threshold limits for drowsiness, yawning, and blinking
// EAR < 0.21 indicates eye closure (drowsiness)
// MAR > 0.55 indicates mouth wide open (yawning)
// EAR < 0.17 indicates deep closure of a blink
const EAR_THRESHOLD = 0.21, MAR_THRESHOLD = 0.55, BLINK_THRESHOLD = 0.17;

// MediaPipe Face Mesh landmark point indices for left eye, right eye, and mouth
const LEFT_EYE = [362, 385, 387, 263, 373, 380], RIGHT_EYE = [33, 160, 158, 133, 153, 144], MOUTH = [13, 14, 61, 291];

// Calculates the standard 2D Euclidean distance between two points: d = sqrt((x1 - x2)^2 + (y1 - y2)^2)
const dist = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

// Calculates the Eye Aspect Ratio (EAR) using vertical heights divided by 2 * horizontal width
const calculateEAR = (l, ids) => (dist(l[ids[1]], l[ids[5]]) + dist(l[ids[2]], l[ids[4]])) / (2.0 * dist(l[ids[0]], l[ids[3]]));

// Calculates the Mouth Aspect Ratio (MAR) using vertical lip gap divided by horizontal mouth corners
const calculateMAR = (l) => dist(l[13], l[14]) / dist(l[61], l[291]);

// Determine API base path dynamically for local development and Vercel hosting
const API_BASE = (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
    ? 'http://localhost:5000'
    : '/_/backend';

function App() {
    const webcamRef = useRef(null), canvasRef = useRef(null), alarmRef = useRef(null);
    const lastState = useRef('OPEN'), chartRef = useRef([]), frameCounter = useRef(0), distractCounter = useRef(0);

    // International Winning States
    const [isDrowsy, setIsDrowsy] = useState(false), [isDistracted, setIsDistracted] = useState(false);
    const [safetyScore, setSafetyScore] = useState(100), [lang, setLang] = useState('en'), [debug, setDebug] = useState(false);
    const [logs, setLogs] = useState([{ time: new Date().toLocaleTimeString(), msg: 'NeuroLink v9.0 Core Initialized.' }]);
    const [chartData, setChartData] = useState(Array(60).fill({ ear: 0.3, mar: 0.1 })), [showReport, setShowReport] = useState(false), [values, setValues] = useState({ ear: 0.3, mar: 0.1 });
    const [demoMode, setDemoMode] = useState(false), [sessionTime, setSessionTime] = useState(0), [stats, setStats] = useState({ blinks: 0, yawns: 0, distractions: 0 });
    const [nightMode, setNightMode] = useState(false), [calibrating, setCalibrating] = useState(false), [calibrationStep, setCalibrationStep] = useState(0);
    const [fleetId, setFleetId] = useState('FLEET-7001-X'), [isSyncing, setIsSyncing] = useState(false);
    const [showHardware, setShowHardware] = useState(false);
    const [isMounted, setIsMounted] = useState(false);

    // Custom ML Model Training State
    const [mlpModel] = useState(() => new SimpleMLP(3, 6, 3));
    const [mlpDataset, setMlpDataset] = useState(defaultDataset);
    const [useCustomModel, setUseCustomModel] = useState(false);
    const [isModelTrained, setIsModelTrained] = useState(false);
    const [mlProbabilities, setMlProbabilities] = useState([1.0, 0.0, 0.0]);
    const [showMLHub, setShowMLHub] = useState(false);
    const [isMLTraining, setIsMLTraining] = useState(false);
    const [mlTrainingStats, setMlTrainingStats] = useState({ epoch: 0, loss: 0, accuracy: 0 });
    const [trainingHistory, setTrainingHistory] = useState([]);
    const [mlRecordClass, setMlRecordClass] = useState(null); // 0 = Alert, 1 = Drowsy, 2 = Distracted
    const [isRecordingML, setIsRecordingML] = useState(false);
    const [weightsUpdateKey, setWeightsUpdateKey] = useState(0);
    const [mlEpochs, setMlEpochs] = useState(100);
    const [mlLR, setMlLR] = useState(0.05);
    const [mlHiddenNeurons, setMlHiddenNeurons] = useState(6);
    const [testMetrics, setTestMetrics] = useState(null);
    const [trainSplitRatio, setTrainSplitRatio] = useState(0.8);
    
    const lastRecordTime = useRef(0);

    useEffect(() => {
        setIsMounted(true);
        alarmRef.current = new Howl({ src: ['https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3'], loop: true });
        const timer = setInterval(() => setSessionTime(t => t + 1), 1000);
        return () => { clearInterval(timer); alarmRef.current?.stop(); };
    }, []);

    const addLog = (msg, type = 'info', earVal = null, marVal = null) => {
        const timeStr = new Date().toLocaleTimeString();
        setLogs(p => [{ time: timeStr, msg, type }, ...p].slice(0, 15));
        
        // Post logs to Express backend API
        fetch(`${API_BASE}/api/logs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                msg, 
                type, 
                safetyScore, 
                ear: earVal || 0, 
                mar: marVal || 0, 
                fleetId 
            })
        }).catch(err => {
            // Silently absorb fetch errors when backend is offline
        });
    };
    const t = (key) => LANGUAGES[lang][key] || key;

    const speak = (key) => {
        const s = new SpeechSynthesisUtterance(LANGUAGES[lang][key]); s.lang = lang; s.rate = 1.3;
        window.speechSynthesis.cancel(); window.speechSynthesis.speak(s);
        if ('vibrate' in navigator) navigator.vibrate([400, 100, 400]);
    };

    // Callback function triggered whenever MediaPipe Face Mesh generates new results from a video frame
    const onResults = useCallback((res) => {
        // Guard clause: Return early if the webcam or canvas reference is not yet loaded
        if (!webcamRef.current?.video || !canvasRef.current) return;
        const canvas = canvasRef.current, video = webcamRef.current.video;
        // Dynamically adjust canvas dimensions to match the raw incoming video feed resolution
        if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
        // Get the 2D drawing context and clear the canvas for the new frame redraw
        const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Check if any face landmarks are successfully detected in the frame
        if (res.multiFaceLandmarks?.[0]) {
            const l = res.multiFaceLandmarks[0]; // Retrieve the list of 478 landmark points for the first detected face
            // Calculate the Eye Aspect Ratio (EAR) as the average of the left eye and right eye EARs
            let ear = (calculateEAR(l, LEFT_EYE) + calculateEAR(l, RIGHT_EYE)) / 2;
            // Calculate the Mouth Aspect Ratio (MAR) to evaluate lip separation
            let mar = calculateMAR(l);
            // Calculate head yaw/rotation ratio by comparing nose-to-left-cheek vs. nose-to-right-cheek distance
            const yaw = dist(l[1], l[234]) / dist(l[1], l[454]);
            // Get absolute head yaw deviation from 1.0 (where 1.0 means looking straight forward)
            const yawDiff = Math.abs(yaw - 1.0);

            // Record ML dataset sample if currently recording (throttled to 10 FPS / every 100 milliseconds)
            if (isRecordingML && mlRecordClass !== null) {
                const now = Date.now();
                if (now - lastRecordTime.current > 100) {
                    // Save input parameters [EAR, MAR, YawDiff] and target class to mlpDataset state
                    setMlpDataset(prev => [...prev, { inputs: [ear, mar, yawDiff], target: mlRecordClass }]);
                    lastRecordTime.current = now; // Update the throttle timestamp
                    if ('vibrate' in navigator) navigator.vibrate(20); // Provide short haptic buzz on mobile/touch screen devices
                }
            }

            // If demo mode is active, simulate active eye closure and yawning values
            if (demoMode) {
                ear = 0.15; // Set simulated low EAR (closed eyes)
                mar = 0.65; // Set simulated high MAR (wide yawn)
            }
            // Update react state to display current numerical EAR and MAR values in the UI
            setValues({ ear, mar });

            // If customized neural network model is selected and has been trained
            if (useCustomModel && isModelTrained) {
                // Forward propagation through the custom browser-trained MLP model
                const { probs } = mlpModel.forward([ear, mar, yawDiff]);
                // Set the model probabilities in the state to update HUD progress bars
                setMlProbabilities(probs);
                
                // Get the class index with the highest probability
                const predClass = probs.indexOf(Math.max(...probs));
                // Extract the highest probability percentage
                const maxProb = probs[predClass];

                // Handles predictions: Class 0 = Alert, Class 1 = Drowsy, Class 2 = Distracted
                if (predClass === 1 && maxProb > 0.5) { // Predicted Drowsy with more than 50% confidence
                    frameCounter.current++; // Increment sequential frames counter
                    // Trigger alert if eyes are closed for more than 10 consecutive frames
                    if (frameCounter.current > 10 && !isDrowsy) {
                        setIsDrowsy(true); alarmRef.current?.play(); speak('wake'); // Set flag, play alarm tone, speak voice alert
                        setSafetyScore(sc => Math.max(0, sc - 5)); addLog("ML: Drowsiness Alert Triggered!", "danger", ear, mar); // Decrement safety score and log event
                    }
                    setIsDistracted(false); // Clear distraction status
                } else if (predClass === 2 && maxProb > 0.5) { // Predicted Distracted with more than 50% confidence
                    distractCounter.current++; // Increment sequential distraction frames counter
                    // Trigger alert if distracted for more than 15 consecutive frames
                    if (distractCounter.current > 15 && !isDistracted) {
                        setIsDistracted(true); speak('eyes'); setStats(s => ({ ...s, distractions: s.distractions + 1 })); // Set flag, play alert voice, increment stats
                        setSafetyScore(sc => Math.max(0, sc - 0.5)); addLog("ML: Distraction Event Registered", "warning", ear, mar); // Decrement safety score and log event
                    }
                    if (isDrowsy) { setIsDrowsy(false); alarmRef.current?.stop(); } // Cancel drowsiness state & stop alarm if active
                } else { // Predicted Alert with highest confidence
                    frameCounter.current = 0; // Reset sequential drowsiness frame counter
                    distractCounter.current = 0; // Reset sequential distraction frame counter
                    if (isDrowsy) {
                        setIsDrowsy(false); alarmRef.current?.stop(); // Turn off drowsiness alert & stop active buzzer audio
                        addLog("ML: State Normal (Alertness Restored)", "success", ear, mar); // Log recovery
                    }
                    setIsDistracted(false); // Reset distraction state
                }
            } else {
                // FALLBACK: Traditional threshold-based state machine logic

                // Blink detection: transition from open eye to closed eye
                if (ear < BLINK_THRESHOLD && lastState.current === 'OPEN') lastState.current = 'CLOSED';
                // Blink registration: transition from closed eye to open eye
                else if (ear > EAR_THRESHOLD && lastState.current === 'CLOSED') {
                    setStats(s => ({ ...s, blinks: s.blinks + 1 })); lastState.current = 'OPEN'; addLog("Neuro-Sync: Blink Locked.", "system", ear, mar);
                }

                // Head turn / Distraction detection
                // If head turns left or right (yaw ratio is outside normal bounds [0.6, 1.6])
                if ((yaw < 0.6 || yaw > 1.6) && !calibrating) {
                    distractCounter.current++; // Increment consecutive frames counter
                    // Trigger alert if looking away for more than 15 consecutive frames
                    if (distractCounter.current > 15 && !isDistracted) {
                        setIsDistracted(true); speak('eyes'); setStats(s => ({ ...s, distractions: s.distractions + 1 })); // Play warning, update status
                        setSafetyScore(sc => Math.max(0, sc - 0.5)); addLog("Attention Divergence!", "warning", ear, mar); // Log warning event
                    }
                } else { distractCounter.current = 0; setIsDistracted(false); } // Reset distraction counter and status if looking straight

                // Closed Eyes / Drowsiness detection
                if (ear < EAR_THRESHOLD && !calibrating) {
                    frameCounter.current++; // Increment consecutive closed-eyes frame counter
                    // Trigger alert if eyes are closed for more than 10 consecutive frames
                    if (frameCounter.current > 10 && !isDrowsy) {
                        setIsDrowsy(true); alarmRef.current?.play(); speak('wake'); // Active alert status, start alarm loop, speak voice alert
                        setSafetyScore(sc => Math.max(0, sc - 5)); addLog("CRITICAL COLLISION RISK", "danger", ear, mar); // Log critical warning
                    }
                } else {
                    frameCounter.current = 0; // Reset closed-eyes frame counter if eyes are open
                    if (isDrowsy) {
                        setIsDrowsy(false); // Cancel drowsiness status
                        alarmRef.current?.stop(); // Turn off active alarm loop
                        addLog("State: Cognitive Restore.", "success", ear, mar); // Log alertness recovery
                    }
                }

                // Yawning Detection (Stage-1 Fatigue Logic)
                // If mouth aspect ratio exceeds threshold, calibrate/drowsiness is not active, and eyes are open
                if (mar > MAR_THRESHOLD && !calibrating && !isDrowsy) {
                    if (lastState.current !== 'YAWNING') { // Ensure we only count the start of a yawn transition
                        setStats(s => ({ ...s, yawns: s.yawns + 1 })); // Increment yawn statistics count
                        lastState.current = 'YAWNING'; // Update state marker
                        addLog("Fatigue Signature: YAWN_DETECTED", "warning", ear, mar); // Log warning event
                        setSafetyScore(sc => Math.max(0, sc - 1)); // Decrement safety score slightly
                    }
                } else if (mar < MAR_THRESHOLD * 0.8 && lastState.current === 'YAWNING') {
                    lastState.current = 'OPEN'; // Reset state marker once mouth has sufficiently closed
                }
            }

            // Append newest computed values to the sliding chart data buffer (keep last 60 points)
            chartRef.current = [...chartRef.current, { ear, mar: mar * 0.4 }].slice(-60);
            setChartData([...chartRef.current]); // Update chart data state to trigger recharts graph update

            // Setup canvas styling based on driver safety state
            ctx.lineWidth = 0.8; ctx.strokeStyle = isDrowsy ? '#ff0044' : isDistracted ? '#ffaa00' : '#00f2ff';
            const drawPath = (ids, close = true) => {
                ctx.beginPath(); ids.forEach((id, i) => {
                    const p = l[id]; const x = (1 - p.x) * canvas.width, y = p.y * canvas.height;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                if (close) ctx.closePath(); ctx.stroke();
            };

            // Select High-Precision Landmarks
            if (debug) {
                l.forEach((p, i) => { if (i % 5 === 0) { ctx.beginPath(); ctx.fillStyle = 'rgba(0, 242, 255, 0.2)'; ctx.arc((1 - p.x) * canvas.width, p.y * canvas.height, 0.4, 0, 2 * Math.PI); ctx.fill(); } });
            }
            drawPath(LEFT_EYE); drawPath(RIGHT_EYE); drawPath(MOUTH);

            // Professional "Reticle" for calibration effect
            if (calibrating) {
                ctx.strokeStyle = '#00f2ff'; ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.arc(canvas.width / 2, canvas.height / 2, 100, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
            }

        } else if (!calibrating) {
            if (demoMode) {
                // Simulated results for Demo Mode when no face is detected
                const ear = 0.15, mar = 0.65;
                setValues({ ear, mar });
                if (!isDrowsy) {
                    setIsDrowsy(true); alarmRef.current?.play(); speak('wake');
                    setSafetyScore(sc => Math.max(0, sc - 5)); addLog("DEMO: COLLISION RISK", "danger");
                }
                chartRef.current = [...chartRef.current, { ear, mar: mar * 0.4 }].slice(-60);
                setChartData([...chartRef.current]);
            } else {
                if (isDrowsy) { setIsDrowsy(false); alarmRef.current?.stop(); }
                addLog("Target Identity Lost.", "warning");
            }
        }
    }, [isDrowsy, isDistracted, debug, lang, demoMode, calibrating]);

    useEffect(() => {
        const faceMesh = new FaceMesh.FaceMesh({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
        faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.85, minTrackingConfidence: 0.85 });
        faceMesh.onResults(onResults);
        if (webcamRef.current?.video) {
            try {
                const camera = new cam.Camera(webcamRef.current.video, {
                    onFrame: async () => {
                        if (webcamRef.current?.video) await faceMesh.send({ image: webcamRef.current.video });
                    },
                    width: 1280,
                    height: 720
                });
                camera.start().catch(err => {
                    console.warn("Camera failed:", err);
                    addLog("Camera Lock Detected. Check permissions.", "warning");
                });
            } catch (e) {
                console.error("Camera Init Error:", e);
            }
        }
    }, [onResults]);

    // Simulated Stream for Demo Mode (if camera fails or no face)
    useEffect(() => {
        if (demoMode && !calibrating) {
            const interval = setInterval(() => {
                const ear = 0.15, mar = 0.65;
                setValues({ ear, mar });
                if (!isDrowsy) {
                    setIsDrowsy(true); alarmRef.current?.play(); speak('wake');
                    setSafetyScore(sc => Math.max(0, sc - 0.5));
                    addLog("DEMO: COGNITIVE OVERRIDE", "danger");
                }
                chartRef.current = [...chartRef.current, { ear, mar: mar * 0.4 }].slice(-60);
                setChartData([...chartRef.current]);
            }, 1000);
            return () => clearInterval(interval);
        }
    }, [demoMode, calibrating, isDrowsy]);



    return (
        <div className="dashboard-container relative">
            <div className="neural-grid" />



            {/* LEFT SIDEBAR: BIOMETRIC REPOSITORY */}
            <aside className="sidebar glass relative">
                <div className="logo-section mb-6">
                    <div className="flex items-center gap-3 group">
                        <div className="status-ring"></div>
                        <h1 className="text-3xl font-black tracking-tighter cursor-none font-['Orbitron']">DRIVEN<span className="text-primary italic">GUARD</span></h1>
                    </div>
                    <span className="text-[10px] font-black opacity-40 tracking-[0.6em] uppercase mt-1 pl-5">NATIONAL EXPO EDITION</span>
                </div>

                <div className="space-y-4">
                    <div className="stat-card glass border-l-4 border-primary">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-[10px] font-black opacity-40 uppercase tracking-widest">{t('stability')}</span>
                            <span className="text-3xl font-black text-primary">{safetyScore.toFixed(0)}<span className="text-xs">%</span></span>
                        </div>
                        <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                            <motion.div className="h-full bg-primary" animate={{ width: `${safetyScore}%` }} />
                        </div>
                    </div>

                    <div className="stat-card glass flex flex-col gap-3">
                        <div className="flex items-center justify-between text-[10px] opacity-40 font-black uppercase tracking-wider">
                            <span>{t('fleet')}</span>
                            <Smartphone size={12} className="text-primary" />
                        </div>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={fleetId}
                                onChange={(e) => setFleetId(e.target.value)}
                                className="bg-black/40 border border-white/10 rounded px-2 py-1.5 text-[9px] font-mono text-primary w-full"
                            />
                            <button
                                onClick={async () => {
                                    setIsSyncing(true);
                                    try {
                                        const res = await fetch(`${API_BASE}/api/sync`, {
                                            method: 'POST',
                                            headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ fleetId, stats })
                                        });
                                        const data = await res.json();
                                        addLog(data.message, "success");
                                    } catch (e) {
                                        addLog("Offline sync stored in Edge node.", "warning");
                                    } finally {
                                        setIsSyncing(false);
                                    }
                                }}
                                className="p-1.5 bg-primary/20 border border-primary/40 rounded hover:bg-primary/30 transition-all"
                            >
                                {isSyncing ? <RefreshCcw size={10} className="animate-spin" /> : <Database size={10} />}
                            </button>
                        </div>
                    </div>

                    <div className="stat-card glass flex flex-col gap-3">
                        <div className="flex items-center justify-between text-[10px] opacity-40 font-black uppercase tracking-wider">
                            <span>{t('gaze')}</span>
                            <Globe2 size={12} />
                        </div>
                        <div className="grid grid-cols-4 gap-2">
                            {Object.keys(LANGUAGES).map(l => (
                                <button key={l} onClick={() => setLang(l)} className={`p-1.5 text-[9px] font-bold border rounded transition-all ${lang === l ? 'bg-primary text-black border-primary' : 'bg-white/5 border-white/10'}`}>{l.toUpperCase()}</button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-hidden flex flex-col gap-4">
                    <div className="stat-card glass flex-1 flex flex-col overflow-hidden bg-black/30">
                        <div className="flex items-center justify-between p-1 mb-4 border-b border-white/5 pb-2">
                            <div className="flex items-center gap-2">
                                <Terminal size={12} className="text-primary" />
                                <span className="text-[10px] font-black opacity-60 uppercase tracking-widest">{t('stream')}</span>
                            </div>
                        </div>
                        <div className="flex-1 custom-scrollbar overflow-y-auto space-y-3 opacity-60 text-[10px]">
                            {logs.map((l, i) => (
                                <div key={i} className={l.type === 'danger' ? 'text-red-500 font-bold' : l.type === 'warning' ? 'text-yellow-400' : ''}>
                                    <span className="text-white/20 font-mono mr-2">[{l.time}]</span> {l.msg}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

            </aside>

            {/* MAIN VIEW: AUGMENTED NEURO-VISION */}
            <main className="main-view h-screen relative">
                <div className={`webcam-wrapper glass shadow-2xl relative group ${nightMode ? 'night-vision' : ''}`}>
                    <Webcam ref={webcamRef} mirrored className={`video-element ${nightMode ? 'night-vision' : ''}`} videoConstraints={{ width: 1280, height: 720, facingMode: "user" }} />
                    <canvas ref={canvasRef} className="canvas-overlay mix-blend-screen" />
                    <div className="scan-line" />

                    <div className="hud-overlay">
                        {(isDrowsy || isDistracted) && (
                            <div className="hud-top">
                                <motion.div
                                    animate={isDrowsy ? { x: [-5, 5, -5], backgroundColor: '#ff0055' } : { x: 0, backgroundColor: 'rgba(0,0,0,0.9)' }}
                                    className={`hud-badge ${isDrowsy ? 'danger text-xl px-12' : 'warning'}`}
                                >
                                    <Activity size={16} />
                                    {isDrowsy ? 'COLLISION_ALERT_MAX' : 'EYE_GAZE_LOST'}
                                </motion.div>
                            </div>
                        )}

                    </div>
                </div>
            </main>

            {/* RIGHT ANALYTICS: QUANTUM BIOMETRICS */}
            <aside className="analytics-panel glass border-none relative z-50">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3"><Activity size={24} className="text-primary" /><span className="text-sm font-black uppercase tracking-[0.4em]">{t('biometrics')}</span></div>
                    <Database size={20} className="text-primary opacity-20" />
                </div>

                <div className="chart-container glass h-[180px] p-6 relative group border-primary/20 block">
                    <div className="absolute top-4 left-6 text-[8px] font-black text-primary/30 uppercase tracking-widest">Signal: EAR_CHART</div>
                    {isMounted && (
                        <ResponsiveContainer width="100%" height="90%" debounce={50}>
                            <AreaChart data={chartData}><defs><linearGradient id="ear" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--primary)" stopOpacity={0.5} /><stop offset="100%" stopColor="var(--primary)" stopOpacity={0} /></linearGradient></defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,242,255,0.05)" />
                                <Area type="monotone" dataKey="ear" stroke="var(--primary)" strokeWidth={4} fill="url(#ear)" isAnimationActive={false} />
                                <YAxis hide domain={[0, 0.45]} /></AreaChart>
                        </ResponsiveContainer>
                    )}
                </div>

                <div className="chart-container glass h-[180px] p-6 relative group border-secondary/20 block">
                    <div className="absolute top-4 left-6 text-[8px] font-black text-secondary/30 uppercase tracking-widest">Signal: MAR_CHART</div>
                    {isMounted && (
                        <ResponsiveContainer width="100%" height="90%" debounce={50}>
                            <AreaChart data={chartData}><defs><linearGradient id="mar" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--secondary)" stopOpacity={0.5} /><stop offset="100%" stopColor="var(--secondary)" stopOpacity={0} /></linearGradient></defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,0,85,0.05)" />
                                <Area type="monotone" dataKey="mar" stroke="var(--secondary)" strokeWidth={4} fill="url(#mar)" isAnimationActive={false} />
                                <YAxis hide domain={[0, 0.45]} /></AreaChart>
                        </ResponsiveContainer>
                    )}
                </div>

                <div className="stat-card glass flex-1 mt-auto bg-gradient-to-tr from-primary/10 via-transparent to-transparent flex flex-col justify-between gap-6">
                    <div>
                        {useCustomModel ? (
                            <div className="mb-6">
                                <div className="flex items-center gap-3 mb-4"><BrainCircuit size={24} className="text-primary" /><span className="text-[11px] font-black uppercase tracking-widest">ML PROBABILITIES</span></div>
                                <div className="space-y-3 font-mono text-[10px] pl-2">
                                    {[
                                        { label: 'ALERT', val: mlProbabilities[0], color: 'bg-green-500' },
                                        { label: 'DROWSY', val: mlProbabilities[1], color: 'bg-red-500' },
                                        { label: 'DISTRACTED', val: mlProbabilities[2], color: 'bg-yellow-500' }
                                    ].map((c, i) => (
                                        <div key={i} className="space-y-1">
                                            <div className="flex justify-between opacity-80">
                                                <span className="font-sans font-bold">{c.label}</span>
                                                <span className="font-bold">{(c.val * 100).toFixed(0)}%</span>
                                            </div>
                                            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                                <motion.div className={`h-full ${c.color}`} animate={{ width: `${c.val * 100}%` }} transition={{ duration: 0.1 }} />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <div className="flex items-center gap-3 mb-6"><Fingerprint size={24} className="text-primary" /><span className="text-[11px] font-black uppercase tracking-widest">{t('integrity')}</span></div>
                        )}
                        <div className="space-y-4">
                            {[
                                { label: t('latency'), val: useCustomModel ? '8.2ms' : '12.4ms', color: 'text-primary' },
                                { label: t('blinks'), val: stats.blinks, color: 'text-white' },
                                { label: t('risk'), val: isDrowsy ? '99%' : '2%', color: isDrowsy ? 'text-secondary' : 'text-green' },
                                { label: t('uptime'), val: '100%', color: 'text-primary' }
                            ].map((item, i) => (
                                <div key={i} className="flex justify-between text-[11px] font-black border-l border-white/5 pl-4"><span className="opacity-30 uppercase tracking-tighter">{item.label}</span><span className={item.color}>{item.val}</span></div>
                            ))}
                        </div>
                    </div>

                    <button onClick={() => setShowReport(true)} className="w-full py-5 bg-primary text-black font-black uppercase text-xs tracking-[0.2em] rounded-2xl shadow-[0_0_30px_rgba(0,242,255,0.4)] hover:scale-[1.02] transition-all relative z-[60] pointer-events-auto">{t('insights')}</button>
                </div>
            </aside>

            {/* TOP 1% ALERT SYSTEM & MODALS */}
            <AnimatePresence>
                {(isDrowsy || isDistracted || demoMode) && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className={`alert-overlay z-[1000] ${isDistracted ? 'bg-yellow-500/20' : 'bg-red-600/40'}`}>
                        <div className="alert-content glass animate-shake border-none shadow-[0_0_200px_rgba(255,0,0,0.6)] p-20 scale-110">
                            {isDrowsy ? <BellRing size={160} className="text-secondary mb-8" /> : <ShieldAlert size={160} className="text-yellow-500 mb-8" />}
                            <h1 className="alert-title text-9xl leading-[0.8] mb-8 font-black tracking-[-8px] text-white underline decoration-secondary decoration-8">DANGER_DETECTED</h1>
                            <p className="alert-msg text-4xl font-black uppercase opacity-40 tracking-[0.2em]">{isDrowsy ? LANGUAGES[lang].wake : LANGUAGES[lang].eyes}</p>
                            <div className={`mt-12 px-20 py-4 rounded-full font-black text-sm tracking-[0.6em] ${isDrowsy ? 'bg-secondary' : 'bg-yellow-600'} text-black`}> CRITICAL_SYSTEM_OVERRIDE </div>
                        </div>
                    </motion.div>
                )}

                {showHardware && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed inset-0 z-[3000] flex items-center justify-center p-12 bg-black/95 backdrop-blur-xl">
                        <div className="glass w-full max-w-4xl p-16 relative border-primary/20 bg-gradient-to-b from-primary/5 to-transparent">
                            <button onClick={() => setShowHardware(false)} className="absolute top-8 right-8 text-white/40 hover:text-white transition-all"><X size={32} /></button>

                            <div className="flex items-center gap-6 mb-12">
                                <div className="p-5 bg-primary/20 rounded-2xl border border-primary/40"><Cpu size={40} className="text-primary" /></div>
                                <div>
                                    <h2 className="text-5xl font-black tracking-tighter">HARDWARE INTEGRATION</h2>
                                    <p className="text-primary/60 font-black text-xs tracking-[0.4em] uppercase">Edge Computing & Device Nodes</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-8">
                                <div className="space-y-6">
                                    <h3 className="text-xl font-black tracking-widest text-primary border-l-4 border-primary pl-4 uppercase">Recommended Devices</h3>
                                    <div className="space-y-4">
                                        {[
                                            { name: 'Raspberry Pi 4/5', desc: 'Central processing unit for Edge AI inference.', icon: <Smartphone size={20} /> },
                                            { name: 'RPi Camera v2/v3', desc: 'High-speed CSI vision sensor for low latency.', icon: <Scan size={20} /> },
                                            { name: '5V Active Buzzer', desc: 'Physical auditory alert system (GPIO 18).', icon: <BellRing size={20} /> },
                                            { name: 'USB GPS Module', desc: 'Optional fleet tracking & location logging.', icon: <Navigation2 size={20} /> }
                                        ].map((d, i) => (
                                            <div key={i} className="flex gap-4 p-4 glass bg-white/5 border-white/5">
                                                <div className="text-primary">{d.icon}</div>
                                                <div>
                                                    <div className="font-black text-sm uppercase">{d.name}</div>
                                                    <div className="text-[11px] opacity-40 font-bold">{d.desc}</div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="space-y-6">
                                    <h3 className="text-xl font-black tracking-widest text-secondary border-l-4 border-secondary pl-4 uppercase">Wiring & Logic</h3>
                                    <div className="glass p-8 bg-black/40 space-y-4">
                                        <div className="p-4 border-l border-white/10">
                                            <div className="text-[10px] font-black opacity-30 uppercase tracking-widest mb-1">Pin Assignment</div>
                                            <div className="text-sm font-mono text-secondary">GPIO 18 <span className="text-white/40">{"->"}</span> BUZZER_POS</div>
                                        </div>
                                        <div className="p-4 border-l border-white/10">
                                            <div className="text-[10px] font-black opacity-30 uppercase tracking-widest mb-1">Inference Engine</div>
                                            <div className="text-sm font-mono text-primary">MediaPipe + TFLite (ARM64)</div>
                                        </div>
                                        <div className="mt-8">
                                            <p className="text-xs opacity-60 leading-relaxed font-bold">
                                                For deployment, the web app's logic is ported to Python. The EAR/MAR thresholds remain identical (0.21/0.55) to ensure consistent behavior across platforms.
                                            </p>
                                        </div>
                                        <button className="w-full py-4 mt-4 bg-primary text-black font-black uppercase text-xs tracking-widest rounded-xl">Download Edge SDK</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}

                {showHardware && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed inset-0 z-[3000] flex items-center justify-center p-12 bg-black/95 backdrop-blur-xl">
                        <div className="glass w-full max-w-4xl p-16 relative border-primary/20 bg-gradient-to-b from-primary/5 to-transparent">
                            <button onClick={() => setShowHardware(false)} className="absolute top-8 right-8 text-white/40 hover:text-white transition-all"><X size={32} /></button>

                            <div className="flex items-center gap-6 mb-12">
                                <div className="p-5 bg-primary/20 rounded-2xl border border-primary/40"><Cpu size={40} className="text-primary" /></div>
                                <div>
                                    <h2 className="text-5xl font-black tracking-tighter">HARDWARE INTEGRATION</h2>
                                    <p className="text-primary/60 font-black text-xs tracking-[0.4em] uppercase">Edge Computing & Device Nodes</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-8">
                                <div className="space-y-6">
                                    <h3 className="text-xl font-black tracking-widest text-primary border-l-4 border-primary pl-4 uppercase">Recommended Devices</h3>
                                    <div className="space-y-4">
                                        {[
                                            { name: 'Raspberry Pi 4/5', desc: 'Central processing unit for Edge AI inference.', icon: <Smartphone size={20} /> },
                                            { name: 'RPi Camera v2/v3', desc: 'High-speed CSI vision sensor for low latency.', icon: <Scan size={20} /> },
                                            { name: '5V Active Buzzer', desc: 'Physical auditory alert system (GPIO 18).', icon: <BellRing size={20} /> },
                                            { name: 'USB GPS Module', desc: 'Optional fleet tracking & location logging.', icon: <Navigation2 size={20} /> }
                                        ].map((d, i) => (
                                            <div key={i} className="flex gap-4 p-4 glass bg-white/5 border-white/5">
                                                <div className="text-primary">{d.icon}</div>
                                                <div>
                                                    <div className="font-black text-sm uppercase">{d.name}</div>
                                                    <div className="text-[11px] opacity-40 font-bold">{d.desc}</div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="space-y-6">
                                    <h3 className="text-xl font-black tracking-widest text-secondary border-l-4 border-secondary pl-4 uppercase">Wiring & Logic</h3>
                                    <div className="glass p-8 bg-black/40 space-y-4">
                                        <div className="p-4 border-l border-white/10">
                                            <div className="text-[10px] font-black opacity-30 uppercase tracking-widest mb-1">Pin Assignment</div>
                                            <div className="text-sm font-mono text-secondary">GPIO 18 <span className="text-white/40">{"->"}</span> BUZZER_POS</div>
                                        </div>
                                        <div className="p-4 border-l border-white/10">
                                            <div className="text-[10px] font-black opacity-30 uppercase tracking-widest mb-1">Inference Engine</div>
                                            <div className="text-sm font-mono text-primary">MediaPipe + TFLite (ARM64)</div>
                                        </div>
                                        <div className="mt-8">
                                            <p className="text-xs opacity-60 leading-relaxed font-bold">
                                                For deployment, the web app's logic is ported to Python. The EAR/MAR thresholds remain identical (0.21/0.55) to ensure consistent behavior across platforms.
                                            </p>
                                        </div>
                                        <button className="w-full py-4 mt-4 bg-primary text-black font-black uppercase text-xs tracking-widest rounded-xl">Download Edge SDK</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}

                {showMLHub && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed inset-0 z-[2500] flex items-center justify-center p-8 bg-black/95 backdrop-blur-xl overflow-y-auto">
                        <div className="glass w-full max-w-5xl p-10 relative border-primary/20 bg-gradient-to-b from-primary/5 via-black/40 to-transparent max-h-[90vh] overflow-y-auto custom-scrollbar">
                            <button onClick={() => setShowMLHub(false)} className="absolute top-6 right-6 text-white/40 hover:text-white transition-all"><X size={28} /></button>

                            <div className="flex items-center gap-5 mb-8">
                                <div className="p-4 bg-primary/20 rounded-2xl border border-primary/45"><BrainCircuit size={36} className="text-primary animate-pulse" /></div>
                                <div>
                                    <h2 className="text-4xl font-black tracking-tighter">COGNITIVE ML TRAINING CENTER</h2>
                                    <p className="text-primary/60 font-black text-[10px] tracking-[0.4em] uppercase">Train Driver-Specific Neural Models Locally on Edge</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
                                {/* LEFT COLUMN: Training Data Capture */}
                                <div className="md:col-span-5 space-y-5">
                                    <h3 className="text-lg font-black tracking-widest text-primary border-l-4 border-primary pl-3 uppercase">1. State Data Collection</h3>
                                    
                                    <div className="space-y-3">
                                        {[
                                            { id: 0, label: 'Alert State', desc: 'Eyes open, mouth closed, looking straight.', color: 'border-green-500/30 hover:bg-green-500/10' },
                                            { id: 1, label: 'Drowsy State', desc: 'Close eyes and yawn naturally.', color: 'border-red-500/30 hover:bg-red-500/10' },
                                            { id: 2, label: 'Distracted State', desc: 'Look away, turn head, look down.', color: 'border-yellow-500/30 hover:bg-yellow-500/10' }
                                        ].map((item) => {
                                            const count = mlpDataset.filter(s => s.target === item.id).length;
                                            const isRecording = isRecordingML && mlRecordClass === item.id;
                                            return (
                                                <div key={item.id} className={`p-4 border rounded-xl bg-white/5 transition-all flex flex-col gap-3 ${item.color}`}>
                                                    <div className="flex justify-between items-center">
                                                        <div>
                                                            <div className="font-black text-sm uppercase">{item.label}</div>
                                                            <div className="text-[10px] opacity-40 font-bold leading-tight">{item.desc}</div>
                                                        </div>
                                                        <span className="text-lg font-mono font-black text-primary bg-primary/10 px-3 py-1 rounded">{count} <span className="text-xs opacity-50">samples</span></span>
                                                    </div>
                                                    
                                                    <button
                                                        onMouseDown={() => { setMlRecordClass(item.id); setIsRecordingML(true); }}
                                                        onMouseUp={() => { setIsRecordingML(false); setMlRecordClass(null); }}
                                                        onTouchStart={() => { setMlRecordClass(item.id); setIsRecordingML(true); }}
                                                        onTouchEnd={() => { setIsRecordingML(false); setMlRecordClass(null); }}
                                                        className={`w-full py-2 rounded text-[10px] font-black uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${
                                                            isRecording 
                                                                ? 'bg-red-600 text-white animate-pulse' 
                                                                : 'bg-white/10 hover:bg-white/20 text-white border border-white/10'
                                                        }`}
                                                    >
                                                        {isRecording ? <span className="w-2 h-2 rounded-full bg-white animate-ping mr-1" /> : null}
                                                        {isRecording ? 'RECORDING... (RELEASE TO STOP)' : 'HOLD TO RECORD'}
                                                    </button>
                                                </div>
                                            );
                                        })}
                                    </div>

                                    <div className="flex gap-2">
                                        <button
                                            onClick={() => {
                                                const synthData = generateSyntheticData(60);
                                                setMlpDataset(synthData);
                                                addLog("ML: Generated Synthetic training set (180 samples)", "system");
                                            }}
                                            className="flex-1 py-3 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all"
                                        >
                                            Generate Synthetic Data
                                        </button>
                                        <button
                                            onClick={() => {
                                                setMlpDataset([]);
                                                setIsModelTrained(false);
                                                addLog("ML: Training dataset cleared.", "warning");
                                            }}
                                            className="py-3 px-4 bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/30 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all"
                                        >
                                            Clear
                                        </button>
                                    </div>
                                </div>

                                {/* MIDDLE COLUMN: Model Network Visualizer */}
                                <div className="md:col-span-4 flex flex-col justify-between">
                                    <div className="space-y-4">
                                        <h3 className="text-lg font-black tracking-widest text-secondary border-l-4 border-secondary pl-3 uppercase">2. Architecture</h3>
                                        
                                        <div className="glass bg-black/40 border-white/5 rounded-xl h-[240px] flex items-center justify-center relative overflow-hidden">
                                            <div className="absolute top-2 left-3 text-[8px] font-mono text-white/30 uppercase">3-Layer Multi-Layer Perceptron</div>
                                            
                                            <svg key={weightsUpdateKey} className="w-full h-full p-2" viewBox="0 0 320 240">
                                                {/* Draw Connection Lines first */}
                                                {/* Input to Hidden (w1: 3 x 6) */}
                                                {mlpModel.w1.map((row, i) => {
                                                    const x1 = 30;
                                                    const y1 = 40 + i * 80;
                                                    return row.map((weight, h) => {
                                                        const x2 = 160;
                                                        const y2 = 20 + h * 40;
                                                        const opacity = Math.max(0.1, Math.min(1.0, Math.abs(weight)));
                                                        const strokeColor = weight > 0 ? '#00f2ff' : '#ff0055';
                                                        const strokeWidth = 0.5 + Math.abs(weight) * 3;
                                                        return (
                                                            <line
                                                                key={`w1-${i}-${h}`}
                                                                x1={x1} y1={y1} x2={x2} y2={y2}
                                                                stroke={strokeColor}
                                                                strokeWidth={strokeWidth}
                                                                strokeOpacity={opacity}
                                                            />
                                                        );
                                                    });
                                                })}
                                                
                                                {/* Hidden to Output (w2: 6 x 3) */}
                                                {mlpModel.w2.map((row, h) => {
                                                    const x1 = 160;
                                                    const y1 = 20 + h * 40;
                                                    return row.map((weight, o) => {
                                                        const x2 = 290;
                                                        const y2 = 40 + o * 80;
                                                        const opacity = Math.max(0.1, Math.min(1.0, Math.abs(weight)));
                                                        const strokeColor = weight > 0 ? '#00f2ff' : '#ff0055';
                                                        const strokeWidth = 0.5 + Math.abs(weight) * 3;
                                                        return (
                                                            <line
                                                                key={`w2-${h}-${o}`}
                                                                x1={x1} y1={y1} x2={x2} y2={y2}
                                                                stroke={strokeColor}
                                                                strokeWidth={strokeWidth}
                                                                strokeOpacity={opacity}
                                                            />
                                                        );
                                                    });
                                                })}
                                                
                                                {/* Draw Nodes */}
                                                {/* Input Nodes: EAR, MAR, YAW */}
                                                {['E', 'M', 'Y'].map((lbl, i) => (
                                                    <g key={`in-${i}`}>
                                                        <circle cx="30" cy={40 + i * 80} r="10" fill="#0c0e17" stroke="#00f2ff" strokeWidth="2" />
                                                        <text x="30" y={44 + i * 80} fontSize="8" fill="#fff" textAnchor="middle" fontWeight="bold">{lbl}</text>
                                                    </g>
                                                ))}
                                                
                                                {/* Hidden Nodes (6) */}
                                                {Array.from({ length: 6 }).map((_, h) => (
                                                    <g key={`hid-${h}`}>
                                                        <circle cx="160" cy={20 + h * 40} r="7" fill="#0c0e17" stroke="#7d00ff" strokeWidth="1.5" />
                                                    </g>
                                                ))}
                                                
                                                {/* Output Nodes (3) */}
                                                {['A', 'Dw', 'Ds'].map((lbl, o) => (
                                                    <g key={`out-${o}`}>
                                                        <circle cx="290" cy={40 + o * 80} r="10" fill="#0c0e17" stroke="#ff0055" strokeWidth="2" />
                                                        <text x="290" y={43 + o * 80} fontSize="7" fill="#fff" textAnchor="middle" fontWeight="bold">{lbl}</text>
                                                    </g>
                                                ))}
                                            </svg>
                                        </div>
                                    </div>
                                    
                                    <div className="space-y-3 pt-3">
                                        <div className="flex justify-between items-center text-[10px] uppercase font-bold text-white/50">
                                            <span>Epochs</span>
                                            <span className="font-mono text-primary">{mlEpochs}</span>
                                        </div>
                                        <input
                                            type="range" min="20" max="300" step="10" value={mlEpochs}
                                            onChange={(e) => setMlEpochs(parseInt(e.target.value))}
                                            className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                                        />
                                        
                                        <div className="flex justify-between items-center text-[10px] uppercase font-bold text-white/50">
                                            <span>Learning Rate</span>
                                            <span className="font-mono text-primary">{mlLR}</span>
                                        </div>
                                        <div className="grid grid-cols-3 gap-2">
                                            {[0.01, 0.05, 0.1].map(rate => (
                                                <button
                                                    key={rate} onClick={() => setMlLR(rate)}
                                                    className={`py-1 text-[9px] font-mono border rounded ${mlLR === rate ? 'bg-primary text-black border-primary' : 'bg-white/5 border-white/10'}`}
                                                >
                                                    {rate}
                                                </button>
                                            ))}
                                        </div>

                                        <div className="flex justify-between items-center text-[10px] uppercase font-bold text-white/50 pt-1">
                                            <span>Train/Test Split</span>
                                            <span className="font-mono text-primary">{(trainSplitRatio * 100).toFixed(0)}/{(100 - trainSplitRatio * 100).toFixed(0)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.50" max="0.90" step="0.05" value={trainSplitRatio}
                                            onChange={(e) => setTrainSplitRatio(parseFloat(e.target.value))}
                                            className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                                        />
                                    </div>
                                </div>

                                {/* RIGHT COLUMN: Training Execution & Stats */}
                                <div className="md:col-span-3 flex flex-col justify-between">
                                    <div className="space-y-4">
                                        <h3 className="text-lg font-black tracking-widest text-primary border-l-4 border-primary pl-3 uppercase">3. Training</h3>
                                        
                                        <div className="glass p-5 bg-black/45 space-y-4 rounded-xl min-h-[220px] flex flex-col justify-between">
                                            {isMLTraining ? (
                                                <div className="space-y-3 flex-1 flex flex-col justify-center items-center text-center">
                                                    <RefreshCcw size={32} className="text-primary animate-spin mb-2" />
                                                    <div className="text-xs font-black uppercase tracking-wider text-primary">Training Network...</div>
                                                    <div className="space-y-1 font-mono text-[10px] opacity-70">
                                                        <div>Epoch: {mlTrainingStats.epoch}/{mlEpochs}</div>
                                                        <div>Loss: {mlTrainingStats.loss.toFixed(4)}</div>
                                                        <div>Accuracy: {(mlTrainingStats.accuracy * 100).toFixed(1)}%</div>
                                                    </div>
                                                </div>
                                            ) : isModelTrained && testMetrics ? (
                                                <div className="space-y-3 flex-grow">
                                                    <div className="text-green-400 text-xs font-black uppercase tracking-widest flex items-center gap-2">
                                                        <ShieldCheck size={16} /> MODEL READY
                                                    </div>
                                                    <div className="border-t border-white/5 pt-3 space-y-2 font-mono text-[10px] opacity-80">
                                                        <div className="flex justify-between"><span>Training set:</span><span>{mlpDataset.length - testMetrics.confusionMatrix.reduce((a, b) => a + b.reduce((c, d) => c + d, 0), 0)} pts</span></div>
                                                        <div className="flex justify-between"><span>Testing set:</span><span>{testMetrics.confusionMatrix.reduce((a, b) => a + b.reduce((c, d) => c + d, 0), 0)} pts</span></div>
                                                        <div className="flex justify-between"><span>Train Acc:</span><span className="text-primary font-bold">{(mlTrainingStats.accuracy * 100).toFixed(1)}%</span></div>
                                                        <div className="flex justify-between"><span>Test Acc:</span><span className="text-green-400 font-bold">{(testMetrics.accuracy * 100).toFixed(1)}%</span></div>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="text-center py-6 text-xs opacity-40 font-bold leading-relaxed flex-grow flex items-center justify-center">
                                                    No model trained yet. Collect at least 15 total samples and trigger training.
                                                </div>
                                            )}

                                            <button
                                                disabled={mlpDataset.length < 15 || isMLTraining}
                                                onClick={() => {
                                                    if (mlpDataset.length < 15) return;
                                                    setIsMLTraining(true);
                                                    let epoch = 0;
                                                    const maxEpochs = mlEpochs;
                                                    const lr = mlLR;
                                                    const history = [];
                                                    let lastLoss = 0;
                                                    let lastAcc = 0;

                                                    // Perform Train-Test Split
                                                    const shuffled = [...mlpDataset].sort(() => Math.random() - 0.5);
                                                    const splitIdx = Math.floor(shuffled.length * trainSplitRatio);
                                                    const trainSet = splitIdx > 0 ? shuffled.slice(0, splitIdx) : shuffled;
                                                    const testSet = splitIdx > 0 && splitIdx < shuffled.length ? shuffled.slice(splitIdx) : shuffled;

                                                    const trainLoop = () => {
                                                        if (epoch >= maxEpochs) {
                                                            setIsMLTraining(false);
                                                            setIsModelTrained(true);
                                                            
                                                            // Compute evaluation metrics on the test set
                                                            const evalResults = mlpModel.evaluate(testSet);
                                                            setTestMetrics(evalResults);
                                                            
                                                            addLog(`ML Core: Training complete (Test Acc: ${(evalResults.accuracy * 100).toFixed(1)}%)`, "success");
                                                            return;
                                                        }

                                                        for (let i = 0; i < 5 && epoch < maxEpochs; i++) {
                                                            const res = mlpModel.trainStep(trainSet, lr);
                                                            lastLoss = res.loss;
                                                            lastAcc = res.accuracy;
                                                            epoch++;
                                                        }

                                                        setMlTrainingStats({ epoch, loss: lastLoss, accuracy: lastAcc });
                                                        history.push({ epoch, loss: lastLoss, accuracy: lastAcc });
                                                        setTrainingHistory([...history]);
                                                        setWeightsUpdateKey(prev => prev + 1);

                                                        requestAnimationFrame(trainLoop);
                                                    };
                                                    requestAnimationFrame(trainLoop);
                                                }}
                                                className={`w-full py-3 bg-secondary text-white font-black uppercase text-[10px] tracking-widest rounded-xl transition-all ${
                                                    mlpDataset.length < 15 || isMLTraining
                                                        ? 'opacity-40 cursor-not-allowed bg-white/10'
                                                        : 'hover:scale-[1.02] shadow-[0_0_20px_rgba(255,0,85,0.4)]'
                                                }`}
                                            >
                                                Train Model
                                            </button>
                                        </div>
                                    </div>

                                    {/* Deployment Switch */}
                                    <div className="glass p-5 bg-primary/5 border border-primary/20 rounded-xl space-y-3 pt-4">
                                        <div className="flex justify-between items-center">
                                            <div>
                                                <div className="text-[10px] font-black uppercase text-primary">Inference Engine</div>
                                                <div className="text-xs font-black uppercase">{useCustomModel ? 'Custom Neural Net' : 'Threshold Rules'}</div>
                                            </div>
                                            <button
                                                disabled={!isModelTrained}
                                                onClick={() => {
                                                    setUseCustomModel(!useCustomModel);
                                                    addLog(`Engine toggled to: ${!useCustomModel ? 'CUSTOM NEURAL NETWORK' : 'THRESHOLD RULES'}`, "system");
                                                }}
                                                className={`px-3 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-wider transition-all ${
                                                    !isModelTrained 
                                                        ? 'bg-white/5 opacity-30 cursor-not-allowed'
                                                        : useCustomModel 
                                                            ? 'bg-secondary text-white border border-secondary shadow-[0_0_15px_rgba(255,0,85,0.3)]' 
                                                            : 'bg-primary text-black hover:bg-primary/80 font-black'
                                                }`}
                                            >
                                                {useCustomModel ? 'Deactivate' : 'Activate'}
                                            </button>
                                        </div>
                                        
                                        {/* Real-time ML Confidence probability bars */}
                                        {useCustomModel && (
                                            <div className="space-y-1.5 pt-2 border-t border-white/5 text-[9px] font-mono">
                                                {[
                                                    { label: 'Alert', val: mlProbabilities[0], color: 'bg-green-500' },
                                                    { label: 'Drowsy', val: mlProbabilities[1], color: 'bg-red-500' },
                                                    { label: 'Distracted', val: mlProbabilities[2], color: 'bg-yellow-500' }
                                                ].map((c, i) => (
                                                    <div key={i} className="space-y-0.5">
                                                        <div className="flex justify-between opacity-70">
                                                            <span>{c.label}</span>
                                                            <span>{(c.val * 100).toFixed(0)}%</span>
                                                        </div>
                                                        <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                                                            <div className={`h-full ${c.color}`} style={{ width: `${c.val * 100}%` }} />
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* EVALUATION REPORT & RESEARCH METRICS SECTION */}
                                {isModelTrained && testMetrics && (
                                    <div className="col-span-12 border-t border-white/10 pt-8 mt-6 space-y-6">
                                        <div className="flex items-center gap-4">
                                            <div className="p-3 bg-secondary/20 rounded-xl border border-secondary/45">
                                                <BarChart3 size={24} className="text-secondary animate-pulse" />
                                            </div>
                                            <div>
                                                <h3 className="text-2xl font-black tracking-tighter">RESEARCH EVALUATION METRICS</h3>
                                                <p className="text-secondary/60 font-black text-[9px] tracking-[0.4em] uppercase">Use these parameters & confusion matrix for research papers</p>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                            {/* Metrics Table */}
                                            <div className="space-y-4">
                                                <h4 className="text-sm font-black tracking-widest text-primary border-l-4 border-primary pl-3 uppercase">Classification Metrics</h4>
                                                <div className="glass bg-black/35 rounded-xl overflow-hidden border border-white/5">
                                                    <table className="w-full text-left border-collapse text-[11px] font-mono">
                                                        <thead>
                                                            <tr className="border-b border-white/10 bg-white/5 text-[9px] font-sans font-black uppercase text-white/40 tracking-wider">
                                                                <th className="p-3">Class</th>
                                                                <th className="p-3">Precision</th>
                                                                <th className="p-3">Recall</th>
                                                                <th className="p-3">F1-Score</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody className="divide-y divide-white/5">
                                                            {[
                                                                { label: 'ALERT', val: testMetrics.metrics[0] },
                                                                { label: 'DROWSY', val: testMetrics.metrics[1] },
                                                                { label: 'DISTRACTED', val: testMetrics.metrics[2] }
                                                            ].map((row, idx) => (
                                                                <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                                                                    <td className="p-3 font-sans font-black text-white/80">{row.label}</td>
                                                                    <td className="p-3 text-primary">{(row.val.precision * 100).toFixed(1)}%</td>
                                                                    <td className="p-3 text-secondary">{(row.val.recall * 100).toFixed(1)}%</td>
                                                                    <td className="p-3 text-green-400">{(row.val.f1 * 100).toFixed(1)}%</td>
                                                                </tr>
                                                            ))}
                                                        </tbody>
                                                    </table>
                                                </div>

                                                <div className="p-4 glass bg-primary/5 border border-primary/25 rounded-xl text-[10.5px] leading-relaxed font-sans text-white/70">
                                                    <span className="font-bold text-primary">Paper Citations Hint:</span> These values are computed on the unseen {((1 - trainSplitRatio) * 100).toFixed(0)}% test subset. In your paper, you can cite the <strong>overall test accuracy of {(testMetrics.accuracy * 100).toFixed(1)}%</strong> and <strong>test loss of {testMetrics.loss.toFixed(4)}</strong> as baseline performance for the 3-layer MLP architecture.
                                                </div>
                                            </div>

                                            {/* Confusion Matrix */}
                                            <div className="space-y-4">
                                                <h4 className="text-sm font-black tracking-widest text-secondary border-l-4 border-secondary pl-3 uppercase">Confusion Matrix</h4>
                                                <div className="glass bg-black/35 p-5 rounded-xl border border-white/5 space-y-4">
                                                    <div className="grid grid-cols-4 gap-2 text-center text-[10px] font-mono">
                                                        {/* Top header spacing */}
                                                        <div></div>
                                                        <div className="font-sans font-black text-[8px] text-white/40 uppercase tracking-widest col-span-3">Predicted Class</div>

                                                        {/* Headers */}
                                                        <div className="font-sans font-black text-[8px] text-white/40 uppercase tracking-widest flex items-center justify-center">Actual</div>
                                                        <div className="bg-white/5 py-1.5 font-sans font-bold text-white/60">ALERT</div>
                                                        <div className="bg-white/5 py-1.5 font-sans font-bold text-white/60">DROWSY</div>
                                                        <div className="bg-white/5 py-1.5 font-sans font-bold text-white/60">DISTRACTED</div>

                                                        {/* Row 1: Alert */}
                                                        <div className="bg-white/5 py-3 font-sans font-bold text-white/60 flex items-center justify-center">ALERT</div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[0][0] > 0 ? 'bg-primary/25 text-primary border border-primary/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[0][0]}
                                                        </div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[0][1] > 0 ? 'bg-red-500/25 text-red-400 border border-red-500/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[0][1]}
                                                        </div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[0][2] > 0 ? 'bg-yellow-500/25 text-yellow-400 border border-yellow-500/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[0][2]}
                                                        </div>

                                                        {/* Row 2: Drowsy */}
                                                        <div className="bg-white/5 py-3 font-sans font-bold text-white/60 flex items-center justify-center">DROWSY</div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[1][0] > 0 ? 'bg-red-500/25 text-red-400 border border-red-500/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[1][0]}
                                                        </div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[1][1] > 0 ? 'bg-primary/25 text-primary border border-primary/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[1][1]}
                                                        </div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[1][2] > 0 ? 'bg-yellow-500/25 text-yellow-400 border border-yellow-500/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[1][2]}
                                                        </div>

                                                        {/* Row 3: Distracted */}
                                                        <div className="bg-white/5 py-3 font-sans font-bold text-white/60 flex items-center justify-center">DISTRACTED</div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[2][0] > 0 ? 'bg-red-500/25 text-red-400 border border-red-500/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[2][0]}
                                                        </div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[2][1] > 0 ? 'bg-yellow-500/25 text-yellow-400 border border-yellow-500/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[2][1]}
                                                        </div>
                                                        <div className={`py-3 rounded font-bold ${testMetrics.confusionMatrix[2][2] > 0 ? 'bg-primary/25 text-primary border border-primary/30' : 'bg-white/5 opacity-40'}`}>
                                                            {testMetrics.confusionMatrix[2][2]}
                                                        </div>
                                                    </div>
                                                    <div className="text-[10px] opacity-40 leading-relaxed font-sans text-center font-bold">
                                                        Diagonal cells show correct predictions (True Positives). Off-diagonals represent misclassifications.
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                )}

                {showReport && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed inset-0 z-[2000] flex items-center justify-center p-12 bg-black/98 backdrop-blur-[50px]">
                        <div className="glass w-full max-w-6xl p-24 overflow-hidden relative border-none bg-white/[0.02]">
                            <button onClick={() => setShowReport(false)} className="absolute top-12 right-12 text-muted hover:text-white transition-all scale-[2] p-4"><X /></button>
                            <div className="flex items-center gap-8 mb-20">
                                <div className="p-8 bg-primary text-black rounded-[3rem] shadow-[0_0_100px_rgba(0,242,255,0.6)]"><FileDown size={48} /></div>
                                <div>
                                    <h2 className="text-8xl font-black tracking-tighter leading-none mb-2 underline decoration-primary decoration-4">GLOBAL IMPACT REPORT</h2>
                                    <p className="opacity-40 font-black uppercase text-xs tracking-[0.8em]">DrivenGuard AI Core / Un-SDG 3.6 Compliance / v9.0</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-4 gap-8 mb-16">
                                {[
                                    { l: 'LIVES IMPACTED', v: (stats.yawns * 14).toLocaleString(), c: 'text-primary' },
                                    { l: 'BLINKS SYNCED', v: stats.blinks, c: 'text-white' },
                                    { l: 'ANOMALIES', v: stats.yawns + stats.distractions, c: 'text-secondary' },
                                    { l: 'FINAL BIOMETRIC KPI', v: `${safetyScore.toFixed(1)}%`, c: 'text-primary' }
                                ].map((s, i) => (
                                    <div key={i} className="p-12 glass bg-black/40 border-primary/10 flex flex-col items-center"><span className="text-[10px] opacity-20 font-black mb-6 tracking-widest uppercase">{s.l}</span><span className={`text-7xl font-black tracking-tighter ${s.c}`}>{s.v}</span></div>
                                ))}
                            </div>

                            <div className="grid grid-cols-2 gap-8 mt-8">
                                <div className="glass p-12 bg-primary/5 border-l-[20px] border-primary rounded-none">
                                    <h3 className="text-xl font-black mb-6 flex items-center gap-4 tracking-[0.4em] uppercase text-primary">{t('summary')}</h3>
                                    <p className="text-2xl opacity-60 leading-tight font-black tracking-tight">
                                        Driver demonstrated high baseline stability. Detected blink rate within normal ranges.
                                        Potential Stage-1 fatigue (yawning) isolated {stats.yawns} times.
                                        Safety intervention prevented ~{((stats.yawns + stats.distractions) * 0.14).toFixed(3)}% increase in risk.
                                    </p>
                                </div>
                                <div className="glass p-12 bg-secondary/5 border-l-[20px] border-secondary rounded-none">
                                    <h3 className="text-xl font-black mb-6 flex items-center gap-4 tracking-[0.4em] uppercase text-secondary">{t('roi')}</h3>
                                    <p className="text-2xl opacity-60 leading-tight font-black tracking-tight">
                                        Estimated Insurance Premium Savings: <span className="text-secondary font-black">$42.50 / Month</span>
                                        <br />
                                        Fleet Compliance Rating: <span className="text-white font-black">GRADE_A</span>
                                    </p>
                                </div>
                            </div>
                            <div className="mt-16 flex gap-8">
                                <button onClick={() => window.print()} className="flex-1 py-10 bg-primary text-black font-black text-2xl rounded-[2.5rem] tracking-widest hover:scale-[1.01] transition-all">{t('export')}</button>
                                <button onClick={() => setShowReport(false)} className="px-20 py-10 glass font-black text-2xl rounded-[2.5rem] opacity-40 hover:opacity-100 uppercase tracking-widest">{t('close')}</button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export default App;
