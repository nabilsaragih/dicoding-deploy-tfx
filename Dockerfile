FROM tensorflow/serving:latest

COPY ./serving_model/mental-health-model /models/mental-health-model
COPY ./config /model_config

ENV MODEL_NAME=mental-health-model
ENV MODEL_BASE_PATH=/models
ENV PORT=8080
ENV MONITORING_CONFIG="/model_config/prometheus.config"

EXPOSE 8080

CMD tensorflow_model_server \
    --rest_api_port=${PORT} \
    --rest_api_host=0.0.0.0 \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
    --monitoring_config_file=${MONITORING_CONFIG}
