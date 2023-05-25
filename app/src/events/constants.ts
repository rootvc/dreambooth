export const SMS_COPY: {
  [key: string]: (args: { [k: string]: string }) => string;
} = {
  STARTED: () =>
    "Hey from RootVC! Stay tuned for your AI avatars shortly. This should take about 90 seconds...",
  FINISHED: ({ id }) =>
    `Your AI avatars are ready! Go grab them from the printer :)`,
};

export const KEYS = {
  QUEUE: {
    TRAINING: "queue:training",
  },
};

export const API_ID = "cfxm61b8e7uzcb";
export const BUCKET = "s3://rootvc-photobooth";
export const GIT_REMOTE_URL = "https://github.com/rootvc/dreambooth.git";
export const PRINT_SERVER = "https://print.root.vc";

export const TWILIO = {
  ACCOUNT_SID: process.env.TWILIO_ACCOUNT_SID,
  AUTH_TOKEN: process.env.TWILIO_AUTH_TOKEN,
  PHONE_NUMBER: process.env.TWILIO_PHONE_NUMBER,
};
