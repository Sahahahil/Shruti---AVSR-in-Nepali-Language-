import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class APIClient {
  private axiosInstance: AxiosInstance;

  constructor() {
    this.axiosInstance = axios.create({
      baseURL: API_BASE_URL,
      timeout: 60000, // 60 seconds for large files
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add error interceptor
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        console.error('API Error:', error.message);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Health check endpoint
   */
  async getHealth() {
    try {
      const response = await this.axiosInstance.get('/health');
      return response.data;
    } catch {
      throw new Error('Failed to connect to backend service');
    }
  }

  /**
   * Get API configuration
   */
  async getConfig() {
    try {
      const response = await this.axiosInstance.get('/api/config');
      return response.data;
    } catch {
      throw new Error('Failed to get API configuration');
    }
  }

  /**
   * Tab 1: Real-time AVSR (Audio + Video fusion)
   */
  async uploadForAVSR(file: File, onProgress?: (progress: number) => void) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await this.axiosInstance.post('/api/avsr/realtime', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            onProgress(progress);
          }
        },
      });

      return response.data;
    } catch (err) {
      if (err instanceof AxiosError) {
        throw new Error(err.response?.data?.detail || 'AVSR processing failed');
      }
      throw err;
    }
  }

  /**
   * Tab 2: Real-time VSR + ASR (Separated)
   */
  async uploadForVSR_ASR(file: File, onProgress?: (progress: number) => void) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await this.axiosInstance.post('/api/vsr-asr/realtime', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            onProgress(progress);
          }
        },
      });

      return response.data;
    } catch (err) {
      if (err instanceof AxiosError) {
        throw new Error(err.response?.data?.detail || 'VSR+ASR processing failed');
      }
      throw err;
    }
  }

  /**
   * Tab 3: ASR Only (Video input, audio processing)
   */
  async uploadForASR_Only(file: File, onProgress?: (progress: number) => void) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await this.axiosInstance.post('/api/asr-only/video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            onProgress(progress);
          }
        },
      });

      return response.data;
    } catch (err) {
      if (err instanceof AxiosError) {
        throw new Error(err.response?.data?.detail || 'ASR processing failed');
      }
      throw err;
    }
  }
}

export const apiClient = new APIClient();
