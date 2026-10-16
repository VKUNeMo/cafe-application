/** Same-origin HTTP access; callers preserve endpoint-specific error displays. */
export const api = {
    predict: (formData) => fetch('/predictor/predict', { method: 'POST', body: formData }),
    histories: () => fetch('/histories/default_user'),
    historyDetail: (id) => fetch(`/histories/${encodeURIComponent(id)}/detail`),
};
