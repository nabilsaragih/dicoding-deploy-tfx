FROM tensorflow/serving:latest

COPY ./serving_model/mental-health-model /models/mental-health-model
COPY ./monitoring/prometheus.config /model_config/prometheus.config

ENV MODEL_NAME=mental-health-model
ENV MODEL_BASE_PATH=/models
ENV PORT=8080
ENV MONITORING_CONFIG="/model_config/prometheus.config"

EXPOSE ${PORT}

RUN echo '#!/bin/bash\n\
exec tensorflow_model_server \\\n\
  --rest_api_port=${PORT} \\\n\
  --model_name=${MODEL_NAME} \\\n\
  --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \\\n\
  --monitoring_config_file=${MONITORING_CONFIG} \\\n\
  "$@"' > /usr/bin/tf_serving_entrypoint.sh && chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
