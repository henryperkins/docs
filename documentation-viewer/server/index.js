const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;

const app = express();

// Enable CORS for all routes
app.use(cors());

// Parse JSON bodies
app.use(express.json());

// API Routes
app.get('/api/documentation', async (req, res) => {
    try {
        const data = await fs.readFile(
            path.join(__dirname, 'data', 'documentation.json'),
            'utf8'
        );
        res.json(JSON.parse(data));
    } catch (error) {
        console.error('Error reading documentation:', error);
        res.status(500).json({
            error: 'Failed to load documentation',
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
});

app.get('/api/metrics', async (req, res) => {
    try {
        const data = await fs.readFile(
            path.join(__dirname, 'data', 'metrics.json'),
            'utf8'
        );
        res.json(JSON.parse(data));
    } catch (error) {
        console.error('Error reading metrics:', error);
        res.status(500).json({
            error: 'Failed to load metrics',
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({
        error: 'Internal Server Error',
        details: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Documentation API available at http://localhost:${PORT}/api/documentation`);
    console.log(`Metrics API available at http://localhost:${PORT}/api/metrics`);
});
