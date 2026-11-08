OpenCV import fix for Streamlit Community Cloud

1. Replace the existing requirements.txt in your GitHub repository with this one.
2. Add packages.txt to the ROOT of the same repository, beside app.py.
3. Delete the failed Streamlit app and create it again.
4. In Advanced settings, select Python 3.12, then save and deploy.
5. Keep app.py and best.pt unchanged for this dependency-only fix.

If it still fails, open Manage app > Logs and copy the complete traceback, especially the final 5-10 lines after import cv2. The screenshot cuts off the actual exception, so this is a compatibility/dependency fix attempt, not a confirmed diagnosis.
