import Documentation from '../models/Documentation';
import logger from '../utils/logger';
import { validateDocumentation } from '../utils/validation';

export const createDocumentation = async (req, res) => {
  try {
    const validation = validateDocumentation(req.body);
    if (!validation.valid) {
      return res.status(400).json({ 
        error: 'Invalid documentation format',
        details: validation.errors 
      });
    }

    const doc = new Documentation(req.body);
    await doc.save();

    logger.info(`Documentation created for ${req.body.file_path}`);
    res.status(201).json(doc);
  } catch (error) {
    if (error.code === 11000) {
      // Handle duplicate key error
      return res.status(409).json({ 
        error: 'Documentation already exists for this file',
        details: error.keyValue 
      });
    }
    logger.error(`Error creating documentation: ${error}`);
    res.status(500).json({ error: 'Error creating documentation' });
  }
};

export const getDocumentation = async (req, res) => {
  try {
    const { project_id, file_path, version } = req.query;
    const query = { project_id };
    
    if (file_path) query.file_path = file_path;
    if (version) query.version = version;

    const docs = await Documentation.find(query)
      .select('-__v')
      .sort({ last_updated: -1 });

    if (!docs.length) {
      return res.status(404).json({ error: 'Documentation not found' });
    }

    res.json(docs);
  } catch (error) {
    logger.error(`Error fetching documentation: ${error}`);
    res.status(500).json({ error: 'Error fetching documentation' });
  }
};

export const updateDocumentation = async (req, res) => {
  try {
    const { project_id, file_path } = req.params;
    const validation = validateDocumentation(req.body);
    if (!validation.valid) {
      return res.status(400).json({ 
        error: 'Invalid documentation format',
        details: validation.errors 
      });
    }

    const doc = await Documentation.findOneAndUpdate(
      { project_id, file_path },
      { 
        ...req.body,
        last_updated: Date.now() 
      },
      { new: true, runValidators: true }
    );

    if (!doc) {
      return res.status(404).json({ error: 'Documentation not found' });
    }

    logger.info(`Documentation updated for ${file_path}`);
    res.json(doc);
  } catch (error) {
    logger.error(`Error updating documentation: ${error}`);
    res.status(500).json({ error: 'Error updating documentation' });
  }
};
