const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("api", {
  backendUrl: () =>
    process.env.MATH_BACKEND_URL || "http://127.0.0.1:8000",
  platform: process.platform,
});