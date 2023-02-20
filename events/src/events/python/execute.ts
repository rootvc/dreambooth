import { loadPyodide } from "pyodide";
import { defineFunction } from "../tools";

const pyodide = await loadPyodide();

export default defineFunction(
  "Execute Python Module",
  "dreambooth/python.execute",
  async ({ tools: { run }, event }) => {
    const result = await run("execute python", async () => {
      const { code } = event.data;
      return await pyodide.runPythonAsync(code);
    });
    return result;
  }
);
