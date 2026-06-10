const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL ?? "").replace(/\/$/, "");


export async function detectKeypointSnapshot({ imageBlob, signal }) {
  if (!imageBlob) {
    throw new Error("No snapshot image was provided.");
  }

  const formData = new FormData();
  const file = new File([imageBlob], "keypoint-snapshot.jpg", {
    type: imageBlob.type || "image/jpeg",
  });
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/predict/keypoints`, {
    method: "POST",
    body: formData,
    signal,
  });

  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("Keypoint response was not valid JSON.");
  }

  if (!response.ok || data.status === "error") {
    throw new Error(data?.detail || data?.message || "Keypoint detection failed.");
  }

  return {
    hands: Boolean(data.keypoints?.hands),
    face: Boolean(data.keypoints?.face),
    pose: Boolean(data.keypoints?.pose),
  };
}
