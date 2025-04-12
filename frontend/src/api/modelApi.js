import axios from 'axios';

const API_BASE = 'http://localhost:5000';

export const fetchAvailableModels = async () => {
  const response = await axios.get(`${API_BASE}/available_models`);
  return response.data.models;
};

export const predictWithBestModel = async (features) => {
  const response = await axios.post(`${API_BASE}/predict`, { features });
  return {
    ...response.data,
    model_name: response.data.model
  };
};

export const predictWithSelectedModel = async (modelName, features) => {
  const response = await axios.post(`${API_BASE}/predict_multi`, {
    model_name: modelName,
    features
  });
  return response.data;
};