'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '@/lib/store';
import RealtimeAVSR from './RealtimeAVSR';
import RealtimeVSR_ASR from './RealtimeVSR_ASR';
import RealtimeASR_VideoInput from './RealtimeASR_VideoInput';
import styles from '@/styles/dashboard.module.scss';

type TabType = 'avsr' | 'vsr_asr' | 'asr_only';

interface TabConfig {
  id: TabType;
  label: string;
  description: string;
  component: React.ComponentType;
}

const tabs: TabConfig[] = [
  {
    id: 'avsr',
    label: 'Realtime AVSR',
    description: 'Live webcam and microphone fusion transcription.',
    component: RealtimeAVSR,
  },
  {
    id: 'vsr_asr',
    label: 'Realtime VSR Only',
    description: 'Live webcam lip-reading without microphone input.',
    component: RealtimeVSR_ASR,
  },
  {
    id: 'asr_only',
    label: 'ASR Only',
    description: 'Live character-level wav2vec2 transcription from microphone audio.',
    component: RealtimeASR_VideoInput,
  },
];

const Dashboard: React.FC = () => {
  const { currentTab, setCurrentTab } = useAppStore();
  const [isHydrated, setIsHydrated] = useState(false);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  if (!isHydrated) {
    return <div className={styles.loading}>Loading...</div>;
  }

  const activeTab = tabs.find((tab) => tab.id === currentTab);
  const TabComponent = activeTab?.component;

  return (
    <div className={styles.dashboard}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h1 className={styles.title}>Shruti</h1>
          <p className={styles.subtitle}>Nepali Realtime Speech Recognition</p>
        </div>
      </div>

      <div className={styles.tabsContainer}>
        <div className={styles.tabsList}>
          {tabs.map((tab, index) => (
            <motion.button
              key={tab.id}
              className={`${styles.tab} ${currentTab === tab.id ? styles.active : ''}`}
              onClick={() => setCurrentTab(tab.id)}
              layout
              layoutId={`tab-${tab.id}`}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.3,
                delay: index * 0.1,
              }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <span className={styles.tabLabel}>{tab.label}</span>
              <span className={styles.tabNumber}>{index + 1}</span>
              {currentTab === tab.id && (
                <motion.div
                  className={styles.activeIndicator}
                  layoutId="active-indicator"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.2 }}
                />
              )}
            </motion.button>
          ))}
        </div>

        <div className={styles.tabDescription}>
          <motion.p
            key={`desc-${currentTab}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab?.description}
          </motion.p>
        </div>
      </div>

      <div className={styles.contentContainer}>
        <AnimatePresence mode="wait">
          {TabComponent && (
            <motion.div
              key={currentTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              className={styles.tabContent}
            >
              <TabComponent />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Dashboard;
