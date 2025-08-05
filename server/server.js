// server.js - Camera System Data Processing Server
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const helmet = require('helmet');
const compression = require('compression');
const mongoose = require('mongoose');

const app = express();
const PORT = process.env.PORT || 5000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb+srv://chuong:maihuychuong@cluster0.wcohcgr.mongodb.net/parkinglotdb?retryWrites=true&w=majority&appName=Cluster0';

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// MongoDB Connection
mongoose.connect(MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => {
    console.log('✅ Connected to MongoDB');
})
.catch((error) => {
    console.error('❌ MongoDB connection error:', error);
    process.exit(1);
});

// ===== ENUMS =====
const EventType = ['enter', 'exit'];
const VehicleType = ['CAR_UNDER_9', 'CAR_9_TO_16'];
const LogStatus = ['IN_PROGRESS', 'COMPLETED'];

// ===== SCHEMAS CHỈ CHO CAMERA SYSTEM =====

// 1. VEHICLES - Quản lý thông tin xe
const VehicleSchema = new mongoose.Schema({
    id: { type: String, required: true, unique: true },
    plateNumber: { type: String, required: true, unique: true },
    vehicleType: { type: String, enum: VehicleType, default: 'CAR_UNDER_9' },
    createdAt: { type: Date, default: Date.now },
    // Thông tin bổ sung từ camera
    first_detected_at: { type: Date, default: Date.now },
    last_seen_at: { type: Date, default: Date.now },
    total_detections: { type: Number, default: 1 }
});

// 2. EVENTS - Sự kiện vào/ra từ camera
const EventSchema = new mongoose.Schema({
    id: { type: String, required: true, unique: true },
    camera_id: { type: String, required: true },
    spot_id: { type: String, required: true },
    spot_name: { type: String, required: true },
    event_type: { type: String, enum: EventType, required: true },
    timestamp: { type: Date, required: true },
    plate_text: String,
    plate_confidence: { type: Number, default: 0 },
    vehicle_image: String,    // base64 image từ camera
    plate_image: String,      // base64 image của biển số
    location_name: String,
    vehicle_id: String,       // Link đến vehicle
    processed: { type: Boolean, default: false },
    created_at: { type: Date, default: Date.now }
});

// 3. STATUSES - Trạng thái hiện tại của các slot
const StatusSchema = new mongoose.Schema({
    spot_id: { type: String, required: true, unique: true },
    spot_name: { type: String, required: true },
    camera_id: { type: String, required: true },
    is_occupied: { type: Boolean, required: true },
    enter_time: Date,
    plate_text: String,
    plate_confidence: { type: Number, default: 0 },
    vehicle_id: String,
    last_update: { type: Date, required: true },
    last_event_id: String,
    updated_at: { type: Date, default: Date.now }
});

// 4. VEHICLE_RECOGNITIONS - Kết quả nhận dạng từ AI
const VehicleRecognitionSchema = new mongoose.Schema({
    id: { type: String, required: true, unique: true },
    event_id: { type: String, required: true },
    vehicle_id: String,       // Có thể null nếu chưa match được
    plate_text: String,
    confidence: { type: Number, required: true },
    bbox: {                   // Bounding box của xe trong ảnh
        x: Number,
        y: Number,
        width: Number,
        height: Number
    },
    plate_bbox: {             // Bounding box của biển số
        x: Number,
        y: Number,
        width: Number,
        height: Number
    },
    vehicle_type_detected: { type: String, enum: VehicleType },
    vehicle_type_confidence: Number,
    processed_at: { type: Date, default: Date.now },
    is_verified: { type: Boolean, default: false },
    processing_time_ms: Number
});

// 5. PARKING_LOGS - Phiên đỗ xe (tự động tạo từ events)
const ParkingLogSchema = new mongoose.Schema({
    id: { type: String, required: true, unique: true },
    vehicle_id: { type: String, required: true },
    spot_id: { type: String, required: true },
    spot_name: String,
    timeIn: { type: Date, required: true },
    timeOut: Date,
    duration_minutes: Number,
    status: { type: String, enum: LogStatus, default: 'IN_PROGRESS' },
    entry_event_id: String,
    exit_event_id: String,
    entry_plate_confidence: Number,
    exit_plate_confidence: Number,
    auto_created: { type: Boolean, default: true },
    created_at: { type: Date, default: Date.now }
});

// ===== MODELS =====
const Vehicle = mongoose.model('Vehicle', VehicleSchema);
const Event = mongoose.model('Event', EventSchema);
const Status = mongoose.model('Status', StatusSchema);
const VehicleRecognition = mongoose.model('VehicleRecognition', VehicleRecognitionSchema);
const ParkingLog = mongoose.model('ParkingLog', ParkingLogSchema);

// ===== UTILITY FUNCTIONS =====
const generateId = () => {
    return Date.now().toString() + Math.random().toString(36).substr(2, 9);
};

// Tìm hoặc tạo vehicle từ biển số
const findOrCreateVehicle = async (plateText, vehicleType = 'CAR_UNDER_9') => {
    try {
        let vehicle = await Vehicle.findOne({ plateNumber: plateText });
        
        if (!vehicle) {
            vehicle = new Vehicle({
                id: generateId(),
                plateNumber: plateText,
                vehicleType: vehicleType,
                first_detected_at: new Date(),
                last_seen_at: new Date(),
                total_detections: 1
            });
            await vehicle.save();
            console.log(`🆕 New vehicle created: ${plateText}`);
        } else {
            // Cập nhật thông tin
            vehicle.last_seen_at = new Date();
            vehicle.total_detections += 1;
            await vehicle.save();
        }
        
        return vehicle;
    } catch (error) {
        console.error('Error in findOrCreateVehicle:', error);
        return null;
    }
};

// Xử lý parking log tự động
const handleParkingLog = async (event, vehicle) => {
    try {
        if (event.event_type === 'enter') {
            // Tạo session đỗ xe mới
            const parkingLog = new ParkingLog({
                id: generateId(),
                vehicle_id: vehicle.id,
                spot_id: event.spot_id,
                spot_name: event.spot_name,
                timeIn: event.timestamp,
                entry_event_id: event.id,
                entry_plate_confidence: event.plate_confidence,
                status: 'IN_PROGRESS'
            });
            await parkingLog.save();
            console.log(`🚗 Parking session started: ${vehicle.plateNumber} at ${event.spot_name}`);
            
        } else if (event.event_type === 'exit') {
            // Kết thúc session đỗ xe
            const activeLog = await ParkingLog.findOne({
                vehicle_id: vehicle.id,
                spot_id: event.spot_id,
                status: 'IN_PROGRESS'
            }).sort({ timeIn: -1 });

            if (activeLog) {
                const durationMinutes = Math.ceil((event.timestamp - activeLog.timeIn) / (1000 * 60));
                
                activeLog.timeOut = event.timestamp;
                activeLog.exit_event_id = event.id;
                activeLog.exit_plate_confidence = event.plate_confidence;
                activeLog.duration_minutes = durationMinutes;
                activeLog.status = 'COMPLETED';
                await activeLog.save();
                
                console.log(`🚪 Parking session completed: ${vehicle.plateNumber} - Duration: ${durationMinutes} minutes`);
            }
        }
    } catch (error) {
        console.error('Error handling parking log:', error);
    }
};

// System health tracking
let systemHealth = {
    lastUpdate: new Date(),
    totalEvents: 0,
    totalVehicleRecognitions: 0,
    totalNewVehicles: 0,
    serverStartTime: new Date()
};

// ===== API ENDPOINTS =====

// Health Check
app.get('/api/health', async (req, res) => {
    try {
        await mongoose.connection.db.admin().ping();
        
        const stats = {
            status: 'ok',
            timestamp: new Date().toISOString(),
            uptime: Math.floor(process.uptime()),
            mongodb_status: 'connected',
            systemHealth,
            database_stats: {
                total_events: await Event.countDocuments(),
                total_vehicles: await Vehicle.countDocuments(),
                total_status_records: await Status.countDocuments(),
                total_recognitions: await VehicleRecognition.countDocuments(),
                active_parking_sessions: await ParkingLog.countDocuments({ status: 'IN_PROGRESS' })
            }
        };
        
        res.json(stats);
        
    } catch (error) {
        console.error('Health check error:', error);
        res.status(500).json({
            status: 'error',
            timestamp: new Date().toISOString(),
            error: error.message,
            mongodb_status: 'disconnected'
        });
    }
});

// 1. EVENTS API - Nhận sự kiện từ camera
app.post('/api/events', async (req, res) => {
    try {
        const eventData = req.body;
        
        // Validate required fields
        const requiredFields = ['camera_id', 'spot_id', 'event_type'];
        for (const field of requiredFields) {
            if (!eventData[field]) {
                return res.status(400).json({
                    error: `Missing required field: ${field}`
                });
            }
        }
        
        // Set defaults
        if (!eventData.id) {
            eventData.id = generateId();
        }
        
        if (!eventData.timestamp) {
            eventData.timestamp = new Date();
        } else {
            eventData.timestamp = new Date(eventData.timestamp);
        }

        // Create event
        const event = new Event(eventData);
        await event.save();

        let vehicle = null;
        
        // Xử lý vehicle nếu có plate_text
        if (eventData.plate_text) {
            vehicle = await findOrCreateVehicle(eventData.plate_text);
            if (vehicle) {
                event.vehicle_id = vehicle.id;
                await event.save();
                
                // Xử lý parking log
                await handleParkingLog(event, vehicle);
            }
        }

        // Update status
        const statusUpdate = {
            spot_id: eventData.spot_id,
            spot_name: eventData.spot_name || eventData.spot_id,
            camera_id: eventData.camera_id,
            is_occupied: eventData.event_type === 'enter',
            last_update: eventData.timestamp,
            last_event_id: eventData.id,
            updated_at: new Date()
        };

        if (eventData.event_type === 'enter') {
            statusUpdate.enter_time = eventData.timestamp;
            statusUpdate.plate_text = eventData.plate_text;
            statusUpdate.plate_confidence = eventData.plate_confidence;
            statusUpdate.vehicle_id = vehicle ? vehicle.id : null;
        } else {
            // Clear data when exit
            statusUpdate.enter_time = null;
            statusUpdate.plate_text = null;
            statusUpdate.plate_confidence = 0;
            statusUpdate.vehicle_id = null;
        }

        await Status.findOneAndUpdate(
            { spot_id: eventData.spot_id },
            statusUpdate,
            { upsert: true, new: true }
        );

        systemHealth.totalEvents++;
        systemHealth.lastUpdate = new Date();

        const logMsg = `📅 ${eventData.event_type.toUpperCase()}: ${eventData.spot_name} - ${eventData.plate_text || 'No plate'} (${eventData.plate_confidence || 0}%)`;
        console.log(logMsg);

        res.status(201).json({
            message: 'Event processed successfully',
            id: eventData.id,
            timestamp: eventData.timestamp,
            event_type: eventData.event_type,
            spot_name: eventData.spot_name,
            vehicle_created: vehicle ? 'existing' : (eventData.plate_text ? 'new' : 'none')
        });

    } catch (error) {
        console.error('❌ Error processing event:', error);
        
        if (error.code === 11000) {
            return res.status(409).json({ 
                error: 'Duplicate event ID',
                id: req.body.id
            });
        }
        
        res.status(500).json({ 
            error: 'Internal server error',
            details: error.message 
        });
    }
});

// 2. VEHICLE RECOGNITION API - Kết quả từ AI
app.post('/api/vehicle-recognition', async (req, res) => {
    try {
        const recognitionData = req.body;
        
        if (!recognitionData.id) {
            recognitionData.id = generateId();
        }

        const startTime = Date.now();
        
        // Tìm vehicle nếu có plate_text
        let vehicle = null;
        if (recognitionData.plate_text) {
            vehicle = await findOrCreateVehicle(
                recognitionData.plate_text, 
                recognitionData.vehicle_type_detected
            );
            if (vehicle) {
                recognitionData.vehicle_id = vehicle.id;
            }
        }

        recognitionData.processing_time_ms = Date.now() - startTime;
        
        const recognition = new VehicleRecognition(recognitionData);
        await recognition.save();

        systemHealth.totalVehicleRecognitions++;
        systemHealth.lastUpdate = new Date();

        console.log(`🤖 AI Recognition: ${recognitionData.plate_text || 'No plate'} (${recognitionData.confidence}%) - ${recognitionData.processing_time_ms}ms`);

        res.status(201).json({
            message: 'Vehicle recognition saved successfully',
            id: recognitionData.id,
            vehicle_id: vehicle ? vehicle.id : null,
            confidence: recognitionData.confidence,
            processing_time_ms: recognitionData.processing_time_ms
        });

    } catch (error) {
        console.error('❌ Error saving vehicle recognition:', error);
        res.status(500).json({ 
            error: 'Internal server error',
            details: error.message 
        });
    }
});

// 3. STATUS API - Cập nhật trạng thái slot
app.post('/api/status', async (req, res) => {
    try {
        const statusData = req.body;
        
        if (!statusData.spot_id) {
            return res.status(400).json({ error: 'spot_id is required' });
        }
        
        if (statusData.last_update) {
            statusData.last_update = new Date(statusData.last_update);
        } else {
            statusData.last_update = new Date();
        }
        
        statusData.updated_at = new Date();

        const result = await Status.findOneAndUpdate(
            { spot_id: statusData.spot_id },
            statusData,
            { upsert: true, new: true, runValidators: true }
        );

        systemHealth.lastUpdate = new Date();

        const occupiedText = statusData.is_occupied ? 'OCCUPIED' : 'AVAILABLE';
        console.log(`📊 ${statusData.spot_name || statusData.spot_id} → ${occupiedText}`);

        res.status(201).json({
            message: 'Status updated successfully',
            spot_id: statusData.spot_id,
            is_occupied: statusData.is_occupied,
            timestamp: statusData.last_update
        });

    } catch (error) {
        console.error('❌ Error updating status:', error);
        res.status(500).json({ 
            error: 'Internal server error',
            details: error.message 
        });
    }
});

// 4. GET APIs for monitoring

// Get current status
app.get('/api/status', async (req, res) => {
    try {
        const statuses = await Status.find().sort({ updated_at: -1 });
        
        const summary = {
            timestamp: new Date().toISOString(),
            total_spots: statuses.length,
            occupied_spots: statuses.filter(s => s.is_occupied).length,
            available_spots: statuses.filter(s => !s.is_occupied).length,
            occupancy_rate: 0,
            spots: statuses
        };

        if (summary.total_spots > 0) {
            summary.occupancy_rate = Math.round((summary.occupied_spots / summary.total_spots) * 100);
        }

        res.json(summary);

    } catch (error) {
        console.error('Error fetching status:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get recent events
app.get('/api/events', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 50;
        const events = await Event.find()
            .sort({ timestamp: -1 })
            .limit(limit);

        res.json({
            timestamp: new Date().toISOString(),
            total_events: await Event.countDocuments(),
            returned_events: events.length,
            events: events
        });

    } catch (error) {
        console.error('Error fetching events:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get vehicles
app.get('/api/vehicles', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 50;
        const search = req.query.search;
        
        let query = {};
        if (search) {
            query.plateNumber = { $regex: search, $options: 'i' };
        }

        const vehicles = await Vehicle.find(query)
            .sort({ last_seen_at: -1 })
            .limit(limit);

        res.json({
            timestamp: new Date().toISOString(),
            total_vehicles: await Vehicle.countDocuments(query),
            returned_vehicles: vehicles.length,
            vehicles: vehicles
        });

    } catch (error) {
        console.error('Error fetching vehicles:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get parking logs
app.get('/api/parking-logs', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 50;
        const status = req.query.status;
        
        let query = {};
        if (status) {
            query.status = status;
        }

        const logs = await ParkingLog.find(query)
            .sort({ created_at: -1 })
            .limit(limit);

        res.json({
            timestamp: new Date().toISOString(),
            total_logs: await ParkingLog.countDocuments(query),
            returned_logs: logs.length,
            logs: logs
        });

    } catch (error) {
        console.error('Error fetching parking logs:', error);
        res.status(500).json({ error: error.message });
    }
});

// Dashboard for camera system
app.get('/api/camera-dashboard', async (req, res) => {
    try {
        const [
            totalEvents,
            totalVehicles,
            totalRecognitions,
            activeSessions,
            recentEvents,
            currentStatus
        ] = await Promise.all([
            Event.countDocuments(),
            Vehicle.countDocuments(),
            VehicleRecognition.countDocuments(),
            ParkingLog.countDocuments({ status: 'IN_PROGRESS' }),
            Event.find().sort({ timestamp: -1 }).limit(10),
            Status.find()
        ]);

        const occupiedSpots = currentStatus.filter(s => s.is_occupied).length;

        res.json({
            timestamp: new Date().toISOString(),
            camera_system_summary: {
                total_events: totalEvents,
                total_vehicles: totalVehicles,
                total_recognitions: totalRecognitions,
                active_parking_sessions: activeSessions,
                occupied_spots: occupiedSpots,
                total_spots: currentStatus.length,
                occupancy_rate: currentStatus.length > 0 ? Math.round((occupiedSpots / currentStatus.length) * 100) : 0
            },
            recent_events: recentEvents,
            current_status: currentStatus.slice(0, 20), // Limit for performance
            system_health: systemHealth
        });

    } catch (error) {
        console.error('Error fetching camera dashboard:', error);
        res.status(500).json({ error: error.message });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({
        error: 'Internal server error',
        timestamp: new Date().toISOString()
    });
});

// Handle 404
app.use('*', (req, res) => {
    res.status(404).json({
        error: 'Endpoint not found',
        path: req.originalUrl,
        method: req.method
    });
});

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('\n🛑 Shutting down camera system server...');
    try {
        await mongoose.connection.close();
        console.log('✅ MongoDB connection closed');
        process.exit(0);
    } catch (error) {
        console.error('❌ Error during shutdown:', error);
        process.exit(1);
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`🎥 Camera System Server running on port ${PORT}`);
    console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
    console.log(`📡 Events (Camera): http://localhost:${PORT}/api/events`);
    console.log(`🤖 AI Recognition: http://localhost:${PORT}/api/vehicle-recognition`);
    console.log(`📈 Status Updates: http://localhost:${PORT}/api/status`);
    console.log(`🚗 Vehicles: http://localhost:${PORT}/api/vehicles`);
    console.log(`📋 Camera Dashboard: http://localhost:${PORT}/api/camera-dashboard`);
    console.log(`⏰ Server started at: ${new Date().toISOString()}`);
});

module.exports = app;