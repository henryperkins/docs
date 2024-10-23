import mongoose from 'mongoose';

const MethodSchema = new mongoose.Schema({
  name: { type: String, required: true },
  docstring: { type: String, required: true },
  args: [String],
  async: Boolean,
  complexity: Number,
  type: { type: String, enum: ['instance', 'class', 'static'] },
  decorators: [String]
});

const ClassSchema = new mongoose.Schema({
  name: { type: String, required: true },
  docstring: { type: String, required: true },
  methods: [MethodSchema],
  decorators: [String],
  superclass: String,
  file_path: String,
  line_number: Number
});

const FunctionSchema = new mongoose.Schema({
  name: { type: String, required: true },
  docstring: { type: String, required: true },
  args: [String],
  async: Boolean,
  complexity: Number,
  decorators: [String],
  file_path: String,
  line_number: Number
});

const MetricsSchema = new mongoose.Schema({
  maintainability_index: Number,
  complexity: Number,
  halstead: {
    volume: Number,
    difficulty: Number,
    effort: Number
  },
  function_metrics: {
    type: Map,
    of: {
      complexity: Number,
      cognitive_complexity: Number,
      lines_of_code: Number
    }
  }
});

const DocumentationSchema = new mongoose.Schema({
  project_id: { 
    type: String, 
    required: true,
    index: true 
  },
  file_path: { 
    type: String, 
    required: true 
  },
  version: {
    type: String,
    required: true
  },
  language: {
    type: String,
    required: true
  },
  summary: String,
  classes: [ClassSchema],
  functions: [FunctionSchema],
  metrics: MetricsSchema,
  last_updated: { 
    type: Date, 
    default: Date.now 
  },
  generated_by: String
}, {
  timestamps: true
});

// Create indexes for common queries
DocumentationSchema.index({ project_id: 1, file_path: 1 }, { unique: true });
DocumentationSchema.index({ 'classes.name': 1 });
DocumentationSchema.index({ 'functions.name': 1 });
DocumentationSchema.index({ last_updated: -1 });

const Documentation = mongoose.model('Documentation', DocumentationSchema);
export default Documentation;
