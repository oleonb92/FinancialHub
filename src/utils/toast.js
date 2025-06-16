import { toast } from 'react-toastify';

// Default options for all toasts
const defaultOptions = {
  position: "top-right",
  autoClose: 3000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  progress: undefined,
};

// Success notification
export const showSuccess = (message, options = {}) => {
  toast.success(message, { ...defaultOptions, ...options });
};

// Error notification
export const showError = (message, options = {}) => {
  toast.error(message, { ...defaultOptions, ...options });
};

// Info notification
export const showInfo = (message, options = {}) => {
  toast.info(message, { ...defaultOptions, ...options });
};

// Warning notification
export const showWarning = (message, options = {}) => {
  toast.warning(message, { ...defaultOptions, ...options });
};

// Custom notification
export const showCustom = (message, options = {}) => {
  toast(message, { ...defaultOptions, ...options });
};

// Dismiss all notifications
export const dismissAll = () => {
  toast.dismiss();
}; 