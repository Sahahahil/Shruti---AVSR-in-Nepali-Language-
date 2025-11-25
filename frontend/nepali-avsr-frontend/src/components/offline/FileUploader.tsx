"use client";
import { useState, useEffect, useRef } from "react";

import "@/styles/components/_offline.scss";

const FileUploader = () => {

    const [file,setFile]=useState(null);
    const inputRef = useRef(null);
    return (
        <section className="fileUploadBody">

            <h1>File Upload</h1>
        </section>
    );
};

export default FileUploader;
