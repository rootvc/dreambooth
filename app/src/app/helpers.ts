import { S3Client } from "@aws-sdk/client-s3";
import { Redis } from "ioredis";

export const AWS_REGION = "us-west-2";
export const BUCKET = "rootvc-photobooth";

export const redis = new Redis(process.env.REDIS_URL || "");
export const s3client = new S3Client({ region: AWS_REGION });
