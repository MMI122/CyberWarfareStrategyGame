"""
FastAPI Application Configuration

This module contains the main FastAPI application setup with:
- CORS middleware
- API routes
- WebSocket handlers
- Error handling
- Authentication (optional)
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import Dict, List, Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events"""
    # Startup
    logger.info("Starting Cyber Warfare Strategy Game API...")
    logger.info("Initializing game manager...")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# =============================================================================
# Create FastAPI Application
# =============================================================================

app = FastAPI(
    title="Cyber Warfare Strategy Game API",
    description="""
    API for the Cyber Warfare Strategy Game - a turn-based strategy game 
    featuring AI agents using MinMax and Deep Reinforcement Learning.
    
    ## Features
    - Create and manage game sessions
    - Execute player actions via REST API
    - Real-time updates via WebSocket
    - AI agent integration (MinMax, Deep RL)
    - Game replays and statistics
    
    ## Research Features
    - Compare AI algorithms performance
    - Collect training data
    - Tournament system
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# =============================================================================
# CORS Configuration
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Import and Include Routers
# =============================================================================

from .routes import games, ai, websocket

app.include_router(games.router, prefix="/api/games", tags=["Games"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with welcome message and links"""
    return {
        "message": "Welcome to Cyber Warfare Strategy Game API",
        "version": "1.0.0",
        "documentation": "/api/docs",
        "health": "/health",
        "endpoints": {
            "games": "/api/games",
            "ai": "/api/ai",
            "websocket": "/ws"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "cyber-warfare-game-api",
        "version": "1.0.0"
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )
