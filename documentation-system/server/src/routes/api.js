// server/routes/api.js
import express from 'express';
import { 
  createDocumentation, 
  getDocumentation,
  updateDocumentation 
} from '../controllers/documentationController';

const router = express.Router();

router.post('/documentation', createDocumentation);
router.get('/documentation', getDocumentation);
router.put('/documentation/:project_id/:file_path', updateDocumentation);

export default router;