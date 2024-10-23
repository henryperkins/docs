import Joi from 'joi';

const methodSchema = Joi.object({
  name: Joi.string().required(),
  docstring: Joi.string().required(),
  args: Joi.array().items(Joi.string()),
  async: Joi.boolean(),
  complexity: Joi.number(),
  type: Joi.string().valid('instance', 'class', 'static'),
  decorators: Joi.array().items(Joi.string())
});

const classSchema = Joi.object({
  name: Joi.string().required(),
  docstring: Joi.string().required(),
  methods: Joi.array().items(methodSchema),
  decorators: Joi.array().items(Joi.string()),
  superclass: Joi.string(),
  file_path: Joi.string(),
  line_number: Joi.number()
});

const functionSchema = Joi.object({
  name: Joi.string().required(),
  docstring: Joi.string().required(),
  args: Joi.array().items(Joi.string()),
  async: Joi.boolean(),
  complexity: Joi.number(),
  decorators: Joi.array().items(Joi.string()),
  file_path: Joi.string(),
  line_number: Joi.number()
});

const metricsSchema = Joi.object({
  maintainability_index: Joi.number(),
  complexity: Joi.number(),
  halstead: Joi.object({
    volume: Joi.number(),
    difficulty: Joi.number(),
    effort: Joi.number()
  }),
  function_metrics: Joi.object().pattern(
    Joi.string(),
    Joi.object({
      complexity: Joi.number(),
      cognitive_complexity: Joi.number(),
      lines_of_code: Joi.number()
    })
  )
});

const documentationSchema = Joi.object({
  project_id: Joi.string().required(),
  file_path: Joi.string().required(),
  version: Joi.string().required(),
  language: Joi.string().required(),
  summary: Joi.string(),
  classes: Joi.array().items(classSchema),
  functions: Joi.array().items(functionSchema),
  metrics: metricsSchema,
  generated_by: Joi.string()
});

export const validateDocumentation = (data) => {
  const { error, value } = documentationSchema.validate(data, {
    abortEarly: false,
    allowUnknown: true
  });

  if (error) {
    return {
      valid: false,
      errors: error.details.map(err => ({
        path: err.path.join('.'),
        message: err.message
      }))
    };
  }

  return { valid: true, value };
};