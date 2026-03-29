import { create } from 'zustand';

interface TranscriptionResult {
  transcription?: string;
  asr_transcription?: string;
  vsr_transcription?: string;
  confidence?: number;
  asr_confidence?: number;
  vsr_confidence?: number;
  processing_time?: number;
  mode?: string;
  status?: string;
}

type CurrentTab =
  | 'avsr';
// | 'vsr_only';
// | 'asr_only';

interface ProcessingState {
  isLoading: boolean;
  uploadProgress: number;
  error: string | null;
  result: TranscriptionResult | null;
  currentTab: CurrentTab;
}

interface AppStore extends ProcessingState {
  setLoading: (loading: boolean) => void;
  setProgress: (progress: number) => void;
  setError: (error: string | null) => void;
  setResult: (result: TranscriptionResult | null) => void;
  setCurrentTab: (tab: CurrentTab) => void;
  reset: () => void;
}

const initialState: ProcessingState = {
  isLoading: false,
  uploadProgress: 0,
  error: null,
  result: null,
  currentTab: 'avsr',
};

export const useAppStore = create<AppStore>((set) => ({
  ...initialState,

  setLoading: (loading) => set({ isLoading: loading }),
  setProgress: (progress) => set({ uploadProgress: progress }),
  setError: (error) => set({ error }),
  setResult: (result) => set({ result }),
  setCurrentTab: (tab) => set({ currentTab: tab }),

  reset: () => set(initialState),
}));
