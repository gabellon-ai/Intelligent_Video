"""
Export Service - Generate CSV, PDF reports, and JSON exports for video analysis
"""

import csv
import json
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image as RLImage, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
from PIL import Image
import cv2
import numpy as np


class ExportService:
    """Service for exporting video analysis results in various formats"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for PDF"""
        self.styles.add(ParagraphStyle(
            name='Title2',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1e40af')
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#374151')
        ))
        self.styles.add(ParagraphStyle(
            name='StatLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#6b7280')
        ))
        self.styles.add(ParagraphStyle(
            name='StatValue',
            parent=self.styles['Normal'],
            fontSize=16,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#111827')
        ))
    
    def generate_csv(
        self,
        job_id: str,
        timeline: List[Dict[str, Any]],
        summary: Dict[str, Any],
        video_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate CSV export with frame-by-frame detections
        
        Returns CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header row
        writer.writerow([
            'frame_number',
            'timestamp_seconds',
            'class_name',
            'confidence',
            'bbox_x1',
            'bbox_y1',
            'bbox_x2',
            'bbox_y2'
        ])
        
        # Write detection data
        for frame_data in timeline:
            frame_num = frame_data.get('frame_number', frame_data.get('frame', 0))
            timestamp = frame_data.get('timestamp', 0)
            detections = frame_data.get('detections', [])
            
            if not detections:
                # Write row with no detections
                writer.writerow([frame_num, f"{timestamp:.3f}", '', '', '', '', '', ''])
            else:
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    writer.writerow([
                        frame_num,
                        f"{timestamp:.3f}",
                        det.get('class_name', det.get('class', '')),
                        f"{det.get('confidence', 0):.4f}",
                        f"{bbox[0]:.2f}" if len(bbox) > 0 else '',
                        f"{bbox[1]:.2f}" if len(bbox) > 1 else '',
                        f"{bbox[2]:.2f}" if len(bbox) > 2 else '',
                        f"{bbox[3]:.2f}" if len(bbox) > 3 else ''
                    ])
        
        return output.getvalue()
    
    def generate_json(
        self,
        job_id: str,
        timeline: List[Dict[str, Any]],
        summary: Dict[str, Any],
        video_info: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive JSON export
        
        Returns full analysis data as dictionary
        """
        export_data = {
            'export_info': {
                'job_id': job_id,
                'exported_at': datetime.utcnow().isoformat() + 'Z',
                'version': '1.0'
            },
            'video_info': video_info or {},
            'filename': filename,
            'summary': summary,
            'timeline': timeline,
            'statistics': self._calculate_statistics(timeline, summary)
        }
        
        return export_data
    
    def _calculate_statistics(
        self,
        timeline: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate additional statistics for export"""
        if not timeline:
            return {}
        
        # Per-class statistics
        class_stats = {}
        total_counts = summary.get('total_counts', {})
        
        for cls, count in total_counts.items():
            frames_with_class = sum(
                1 for frame in timeline 
                if cls in frame.get('counts', {})
            )
            class_stats[cls] = {
                'total_detections': count,
                'frames_with_detection': frames_with_class,
                'percentage_of_frames': (frames_with_class / len(timeline) * 100) if timeline else 0
            }
        
        # Timeline density
        detections_per_frame = [
            len(frame.get('detections', []))
            for frame in timeline
        ]
        
        return {
            'class_statistics': class_stats,
            'detection_density': {
                'average_per_frame': sum(detections_per_frame) / len(detections_per_frame) if detections_per_frame else 0,
                'max_per_frame': max(detections_per_frame) if detections_per_frame else 0,
                'min_per_frame': min(detections_per_frame) if detections_per_frame else 0
            },
            'total_frames_analyzed': len(timeline)
        }
    
    def generate_pdf(
        self,
        job_id: str,
        timeline: List[Dict[str, Any]],
        summary: Dict[str, Any],
        video_info: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        video_path: Optional[str] = None
    ) -> bytes:
        """
        Generate comprehensive PDF report
        
        Returns PDF as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Video Analysis Report", self.styles['Title2']))
        story.append(Spacer(1, 12))
        
        # Job Info Header
        story.append(self._create_header_section(job_id, video_info, filename))
        story.append(Spacer(1, 20))
        
        # Summary Statistics
        story.append(Paragraph("Summary Statistics", self.styles['SectionHeader']))
        story.append(self._create_summary_table(summary))
        story.append(Spacer(1, 20))
        
        # Object Counts Chart
        if summary.get('total_counts'):
            story.append(Paragraph("Object Detection Counts", self.styles['SectionHeader']))
            chart = self._create_bar_chart(summary['total_counts'])
            if chart:
                story.append(chart)
            story.append(Spacer(1, 20))
        
        # Timeline Visualization
        if timeline:
            story.append(Paragraph("Detection Timeline", self.styles['SectionHeader']))
            timeline_drawing = self._create_timeline_visualization(timeline, video_info)
            if timeline_drawing:
                story.append(timeline_drawing)
            story.append(Spacer(1, 20))
        
        # Detection Details Table
        story.append(Paragraph("Detection Details by Class", self.styles['SectionHeader']))
        story.append(self._create_detection_details_table(summary, timeline))
        story.append(Spacer(1, 20))
        
        # Key Frames with highest detections
        if video_path and timeline:
            story.append(PageBreak())
            story.append(Paragraph("Key Frames (Highest Detection Counts)", self.styles['SectionHeader']))
            key_frames = self._get_key_frames(timeline, video_path, top_n=4)
            if key_frames:
                story.extend(key_frames)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_header_section(
        self,
        job_id: str,
        video_info: Optional[Dict[str, Any]],
        filename: Optional[str]
    ) -> Table:
        """Create the header info section"""
        now = datetime.now()
        
        duration = video_info.get('duration', 0) if video_info else 0
        fps = video_info.get('fps', 0) if video_info else 0
        resolution = f"{video_info.get('width', 0)}x{video_info.get('height', 0)}" if video_info else 'N/A'
        
        # Format duration
        mins, secs = divmod(int(duration), 60)
        duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        
        data = [
            ['Job ID:', job_id, 'Report Date:', now.strftime('%Y-%m-%d %H:%M')],
            ['Filename:', filename or 'N/A', 'Duration:', duration_str],
            ['Resolution:', resolution, 'Frame Rate:', f"{fps:.1f} FPS" if fps else 'N/A'],
        ]
        
        table = Table(data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6b7280')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#6b7280')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f4f6')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
        ]))
        
        return table
    
    def _create_summary_table(self, summary: Dict[str, Any]) -> Table:
        """Create summary statistics table"""
        total_detections = summary.get('total_detections', 0)
        unique_classes = len(summary.get('unique_classes', []))
        frames_analyzed = summary.get('total_frames_analyzed', 0)
        frames_with_det = summary.get('frames_with_detections', 0)
        
        # Calculate detection rate
        detection_rate = (frames_with_det / frames_analyzed * 100) if frames_analyzed else 0
        
        data = [
            ['Total Detections', 'Unique Classes', 'Frames Analyzed', 'Detection Rate'],
            [str(total_detections), str(unique_classes), str(frames_analyzed), f"{detection_rate:.1f}%"]
        ]
        
        table = Table(data, colWidths=[1.65*inch, 1.65*inch, 1.65*inch, 1.65*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, 1), 18),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#6b7280')),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#1e40af')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff6ff')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bfdbfe')),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#bfdbfe')),
        ]))
        
        return table
    
    def _create_bar_chart(self, total_counts: Dict[str, int]) -> Optional[Drawing]:
        """Create a bar chart of object counts"""
        if not total_counts:
            return None
        
        # Prepare data
        categories = list(total_counts.keys())
        values = list(total_counts.values())
        
        if not values:
            return None
        
        # Create drawing
        width = 500
        height = 200
        drawing = Drawing(width, height)
        
        # Create bar chart
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 30
        chart.width = width - 100
        chart.height = height - 60
        chart.data = [values]
        chart.categoryAxis.categoryNames = categories
        chart.categoryAxis.labels.fontName = 'Helvetica'
        chart.categoryAxis.labels.fontSize = 8
        chart.categoryAxis.labels.angle = 30
        chart.categoryAxis.labels.boxAnchor = 'ne'
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(values) * 1.2
        chart.valueAxis.labels.fontName = 'Helvetica'
        chart.valueAxis.labels.fontSize = 8
        chart.bars[0].fillColor = colors.HexColor('#3b82f6')
        chart.bars[0].strokeColor = colors.HexColor('#1e40af')
        chart.barWidth = min(30, (chart.width / len(categories)) * 0.6)
        
        drawing.add(chart)
        return drawing
    
    def _create_timeline_visualization(
        self,
        timeline: List[Dict[str, Any]],
        video_info: Optional[Dict[str, Any]]
    ) -> Optional[Drawing]:
        """Create a timeline visualization showing detection density"""
        if not timeline:
            return None
        
        width = 500
        height = 80
        drawing = Drawing(width, height)
        
        # Background
        drawing.add(Rect(0, 20, width, 40, fillColor=colors.HexColor('#f3f4f6'), strokeColor=None))
        
        # Calculate detection density
        duration = video_info.get('duration', 0) if video_info else 0
        if not duration:
            duration = max((f.get('timestamp', 0) for f in timeline), default=0) or 1
        
        # Create density bars
        num_segments = min(50, len(timeline))
        segment_width = width / num_segments
        
        # Group timeline into segments
        segment_counts = [0] * num_segments
        for frame in timeline:
            ts = frame.get('timestamp', 0)
            segment_idx = min(int(ts / duration * num_segments), num_segments - 1)
            segment_counts[segment_idx] += len(frame.get('detections', []))
        
        max_count = max(segment_counts) if segment_counts else 1
        
        for i, count in enumerate(segment_counts):
            if count > 0:
                bar_height = (count / max_count) * 36
                intensity = min(255, int(count / max_count * 255))
                color = colors.HexColor(f'#3b82f6')
                drawing.add(Rect(
                    i * segment_width + 1, 22,
                    segment_width - 2, bar_height,
                    fillColor=color,
                    strokeColor=None
                ))
        
        # Time labels
        drawing.add(String(5, 5, "0:00", fontSize=8, fontName='Helvetica', fillColor=colors.HexColor('#6b7280')))
        mins, secs = divmod(int(duration), 60)
        drawing.add(String(width - 30, 5, f"{mins}:{secs:02d}", fontSize=8, fontName='Helvetica', fillColor=colors.HexColor('#6b7280')))
        
        # Title
        drawing.add(String(5, height - 10, "Detection Activity Over Time", fontSize=9, fontName='Helvetica-Bold', fillColor=colors.HexColor('#374151')))
        
        return drawing
    
    def _create_detection_details_table(
        self,
        summary: Dict[str, Any],
        timeline: List[Dict[str, Any]]
    ) -> Table:
        """Create detailed detection table by class"""
        total_counts = summary.get('total_counts', {})
        max_simultaneous = summary.get('max_simultaneous', {})
        
        # Header
        data = [['Class', 'Total Count', 'Max Simultaneous', 'Avg Confidence']]
        
        # Calculate average confidence per class
        class_confidences = {}
        for frame in timeline:
            for det in frame.get('detections', []):
                cls = det.get('class_name', det.get('class', 'unknown'))
                conf = det.get('confidence', 0)
                if cls not in class_confidences:
                    class_confidences[cls] = []
                class_confidences[cls].append(conf)
        
        for cls, count in sorted(total_counts.items(), key=lambda x: -x[1]):
            avg_conf = sum(class_confidences.get(cls, [0])) / len(class_confidences.get(cls, [1]))
            data.append([
                cls,
                str(count),
                str(max_simultaneous.get(cls, 0)),
                f"{avg_conf:.1%}"
            ])
        
        if len(data) == 1:
            data.append(['No detections', '-', '-', '-'])
        
        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#1e3a8a')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ]))
        
        return table
    
    def _get_key_frames(
        self,
        timeline: List[Dict[str, Any]],
        video_path: str,
        top_n: int = 4
    ) -> List:
        """Extract and annotate key frames with highest detection counts"""
        if not timeline or not video_path:
            return []
        
        # Sort frames by detection count
        sorted_frames = sorted(
            timeline,
            key=lambda x: len(x.get('detections', [])),
            reverse=True
        )[:top_n]
        
        if not sorted_frames:
            return []
        
        elements = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return [Paragraph("Could not load video frames", self.styles['Normal'])]
        
        try:
            for i, frame_data in enumerate(sorted_frames):
                frame_num = frame_data.get('frame_number', frame_data.get('frame', 0))
                timestamp = frame_data.get('timestamp', 0)
                detections = frame_data.get('detections', [])
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Draw bounding boxes
                for det in detections:
                    bbox = det.get('bbox', [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = [int(b) for b in bbox]
                        cls = det.get('class_name', det.get('class', 'object'))
                        conf = det.get('confidence', 0)
                        
                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{cls}: {conf:.0%}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 4, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Convert to RGB and resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize to fit in PDF
                max_width = 350
                ratio = max_width / pil_image.width
                new_height = int(pil_image.height * ratio)
                pil_image = pil_image.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to bytes for reportlab
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Create image element
                img = RLImage(img_buffer, width=max_width, height=new_height)
                
                # Frame info
                mins, secs = divmod(int(timestamp), 60)
                counts_str = ", ".join(f"{k}: {v}" for k, v in frame_data.get('counts', {}).items())
                
                frame_info = Paragraph(
                    f"<b>Frame {frame_num}</b> at {mins}:{secs:02d} — {len(detections)} detections ({counts_str})",
                    self.styles['Normal']
                )
                
                elements.append(KeepTogether([frame_info, Spacer(1, 5), img, Spacer(1, 15)]))
        
        finally:
            cap.release()
        
        return elements


# Singleton instance
export_service = ExportService()
