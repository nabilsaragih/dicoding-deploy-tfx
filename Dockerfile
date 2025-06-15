FROM tensorflow/serving:latest

# Copy model and config
COPY ./serving_model/mental-health-model /models/mental-health-model
COPY ./config/prometheus.config /model_config/prometheus.config

# Set environment variables
ENV MODEL_NAME=mental-health-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG=/model_config/prometheus.config

# Expose the port Railway sets (typically 8080)
EXPOSE ${PORT}

# Entrypoint script for TensorFlow Serving
RUN echo '#!/bin/bash\n\
exec tensorflow_model_server \\\n\
  --rest_api_port=${PORT} \\\n\
  --model_name=${MODEL_NAME} \\\n\
  --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \\\n\
  --monitoring_config_file=${MONITORING_CONFIG} \\\n\
  "$@"' > /usr/bin/tf_serving_entrypoint.sh \
  && chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
